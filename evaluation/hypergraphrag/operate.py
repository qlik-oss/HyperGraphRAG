import asyncio
import json
import logging
import re
import copy
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union, Callable, cast
from collections import Counter, defaultdict
import warnings
import igraph as ig
import numpy as np
import math
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    normalize_entity_name,
    min_max_normalize,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results

async def build_entity_to_chunk_ids_map(knowledge_graph_inst: BaseGraphStorage) -> dict[str, set[str]]:
    # Build entity -> set of chunk ids mapping
    SEP_TOKEN = "<SEP>"
    entity_to_chunk_ids = defaultdict(set)
    for node_id, node_data in await knowledge_graph_inst.get_all_nodes():
        if node_data.get("role") == "entity":
            source_ids = node_data.get("source_id", "")
            if source_ids and source_ids != "UNKNOWN":
                chunk_ids = [cid.strip() for cid in source_ids.split(SEP_TOKEN) if cid.strip()]
                entity_to_chunk_ids[node_id].update(chunk_ids)

    return entity_to_chunk_ids

async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary, metadata = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    )
    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        hyper_relation="<hyperedge>"+knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
    )
    

async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        already_weights.append(already_hyperedge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    node_data = dict(
        role = "hyperedge",
        weight=weight,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        hyperedge_name,
        node_data=node_data,
    )
    node_data["hyperedge_name"] = hyperedge_name
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []
    entity_name = normalize_entity_name(entity_name)
    # Check if the node already exists in the knowledge graph
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []
    
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        
        already_weights = []
        already_source_ids = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                weight=weight,
                source_id=source_id,
            ),
        )

        edge_data.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
        ))

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    # Tokens usages statistics
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    num_cache_hit = 0

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        nonlocal total_prompt_tokens, total_completion_tokens, total_cached_tokens, num_cache_hit
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result, metadata = await use_llm_func(hint_prompt)
        total_prompt_tokens += metadata.get('prompt_tokens', 0)
        total_completion_tokens += metadata.get('completion_tokens', 0)
        total_cached_tokens += metadata.get('cached_tokens', 0)
        if metadata.get('cache_hit'):
            num_cache_hit += 1

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            logger.debug(f"gleaning entities for {entity_extract_max_gleaning}")
            logger.debug(f"continue_prompt: {continue_prompt}")
            logger.debug(f"history_messages: {history}")

            glean_result, glean_metadata = await use_llm_func(continue_prompt, history_messages=history)
            total_prompt_tokens += glean_metadata.get('prompt_tokens', 0)
            total_completion_tokens += glean_metadata.get('completion_tokens', 0)
            total_cached_tokens += glean_metadata.get('cached_tokens', 0)

            if glean_metadata.get('cache_hit'):
                num_cache_hit += 1

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            
            logger.debug(f"all history: {history}")
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result, loop_metadata = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            total_prompt_tokens += loop_metadata.get('prompt_tokens', 0)
            total_completion_tokens += loop_metadata.get('completion_tokens', 0)
            total_cached_tokens += loop_metadata.get('cached_tokens', 0)

            if loop_metadata.get('cache_hit'):
                num_cache_hit += 1
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_relation = await _handle_single_hyperrelation_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[if_relation["hyper_relation"]].append(
                    if_relation
                )
                now_hyper_relation = if_relation["hyper_relation"]
                
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, now_hyper_relation
            )
            if if_entities is not None:
                # Check for duplicates before appending
                entity_name = if_entities["entity_name"]
                existing_entities = maybe_nodes[entity_name]

                # Check if an identical or very similar entity already exists
                is_duplicate = False
                for existing in existing_entities:
                    if (existing["hyper_relation"] == if_entities["hyper_relation"] and
                        existing["description"].lower() == if_entities["description"].lower()):
                        # Exact duplicate found, skip it
                        is_duplicate = True
                        break

                if not is_duplicate:
                    maybe_nodes[entity_name].append(if_entities)
                continue
            
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    process_semaphore = asyncio.Semaphore(8)  # Limit to 8 concurrent tasks

    async def _process_with_limit(chunk):
        async with process_semaphore:
            return await _process_single_content(chunk)
    
    results = []
    progress_bar = tqdm_async(
        asyncio.as_completed([_process_with_limit(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    )
    for result in progress_bar:
        results.append(await result)
        progress_bar.set_postfix({
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'num_cache_hit': num_cache_hit,
            'total_cached_tokens': total_cached_tokens
        })

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
            
    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_hyperedges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting hyperedges",
        unit="entity",
    ):
        all_hyperedges_data.append(await result)
            
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any hyperedges and entities, maybe your LLM is not working"
        )
        return None

    if not len(all_hyperedges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if hyperedge_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
            }
            for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + ' ' + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: BaseVectorStorage,  
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    # Handle cache
    query_param.query = query
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response
    
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )
    
    hint_prompt = entity_extract_prompt.format(
        **context_base, input_text="{input_text}"
    ).format(**context_base, input_text=query)

    final_result = await use_model_func(hint_prompt)

    logger.info("kw_prompt result:")
    print(final_result)
    hl_keywords, ll_keywords = [], []
    try:
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                hl_keywords.append("<hyperedge>"+clean_str(record_attributes[1]))
            elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                ll_keywords.append(clean_str(record_attributes[1]).upper())
            else:
                continue
    # Handle parsing error
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {final_result}")
        return PROMPTS["fail_response"]

    # Handdle keywords missing
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["hybrid","local"]:
        logger.warning("low_level_keywords is empty")
        ll_keywords = query
    else:
        ll_keywords = ", ".join(ll_keywords)
    if hl_keywords == [] and query_param.mode in ["hybrid","global"]:
        logger.warning("high_level_keywords is empty")
        hl_keywords = query
    else:
        hl_keywords = ", ".join(hl_keywords)

    # Only build entity->chunk map when PPR is active (it's the only consumer)
    if query_param.use_ppr_text_units:
        entity_to_chunk_ids = await build_entity_to_chunk_ids_map(knowledge_graph_inst)
    else:
        entity_to_chunk_ids = {}

    # Build context
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        hyperedges_vdb,
        text_chunks_db,
        chunks_vdb,
        query_param,
        entity_to_chunk_ids,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response, metadata = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response


async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
    entity_to_chunk_ids: dict[str, set[str]],
):

    ll_keywords, hl_keywords = query[0], query[1]
    if query_param.mode in ["local", "hybrid"]:
        if ll_keywords == "":
            ll_entities_context, ll_relations_context, ll_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "Low Level context is None. Return empty Low entity/relationship/source"
            )
            query_param.mode = "global"
        else:
            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
            ) = await _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
                entity_to_chunk_ids,
            )
    if query_param.mode in ["global", "hybrid"]:
        if hl_keywords == "":
            hl_entities_context, hl_relations_context, hl_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "High Level context is None. Return empty High entity/relationship/source"
            )
            query_param.mode = "local"
        else:
            (
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
            ) = await _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                hyperedges_vdb,
                text_chunks_db,
                query_param,
                entity_to_chunk_ids,
            )
            if (
                hl_entities_context == ""
                and hl_relations_context == ""
                and hl_text_units_context == ""
            ):
                logger.warn("No high level context found. Switching to local mode.")
                query_param.mode = "local"
    if query_param.mode == "hybrid":
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    elif query_param.mode == "local":
        entities_context, relations_context, text_units_context = (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        )
    
    # PPR optimization enabled, run PPR text units ranking
    if  query_param.ppr_enabled:
        ppr_text_units_context = await _ppr_rank_text_units(
            query_param=query_param,
            knowledge_graph_inst=knowledge_graph_inst,
            text_chunks_db=text_chunks_db,
            chunks_vdb=chunks_vdb,
        )
        text_units_context = ppr_text_units_context if ppr_text_units_context else text_units_context

    return f"""
        -----Entities-----
        ```csv
        {entities_context}
        ```
        -----Relationships-----
        ```csv
        {relations_context}
        ```
        -----Sources-----
        ```csv
        {text_units_context}
        ```
        """


async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    entity_to_chunk_ids: dict[str, set[str]],
):
    # get similar entities
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""
    
    # Normalize entity distances and store similarities only when PPR is active
    if query_param.ppr_enabled:
        # Normalize entity distance by chunk connectivity to avoid bias towards generic entities.
        for r in results:
            chunk_ids = entity_to_chunk_ids.get(r["entity_name"], set())
            chunk_count = len(chunk_ids) if len(chunk_ids) > 0 else 1
            r["distance"] /= chunk_count
            prev_val = query_param._entity_sims.get(r["entity_name"], 0.0)
            query_param._entity_sims[r["entity_name"]] = max(float(r.get("distance", 0.0)), prev_val)

    # get entity information
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # get entity degree
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    # get relate edges
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    # build prompt
    entites_section_list = [["id", "entity", "type", "description"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "hyperedge", "related_entities"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["description"],
                e["related_nodes"]
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(e)
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, "description": k[1], **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    all_related_nodes = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["src_tgt"][1]) for edge in all_edges_data]
    )
    all_nodes = []
    for this_nodes in all_related_nodes:
        all_nodes.append(GRAPH_FIELD_SEP.join([n[1] for n in this_nodes]))
    all_edges_data = [
        {**e, "related_nodes": n}
        for e, n in zip(all_edges_data, all_nodes)
    ]
    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    entity_to_chunk_ids: dict[str, set[str]],
):
    results = await hyperedges_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["hyperedge_name"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    # edge_degree = await asyncio.gather(
    #     *[knowledge_graph_inst.node_degree(r["hyperedge_name"]) for r in results]
    # )
    edge_datas = [
        {"hyperedge": k["hyperedge_name"], "rank": k["distance"], **v}
        for k, v in zip(results, edge_datas)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    if query_param.ppr_enabled:
        query_param._hyperedge_sims.update({e["hyperedge"]: e["rank"] for e in edge_datas})
    
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["hyperedge"],
        max_token_size=query_param.max_token_for_global_context,
    )
    all_related_nodes = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas]
    )
    all_nodes = []
    for this_nodes in all_related_nodes:
        all_nodes.append(GRAPH_FIELD_SEP.join([n[1] for n in this_nodes]))
    edge_datas = [
        {**e, "related_nodes": n}
        for e, n in zip(edge_datas, all_nodes)
    ]

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )

    # Propagate hyperedge similarity to connected entities — only needed for PPR
    if query_param.ppr_enabled:
        entity_hyperedge_counts = defaultdict(int)
        entity_hyperedge_scores = defaultdict(float)
        for edge in edge_datas:
            hyperedge_name = edge["hyperedge"]
            hyperedge_score = query_param._hyperedge_sims.get(hyperedge_name, 1.0)
            for entity_name in edge['related_nodes'].split(GRAPH_FIELD_SEP):
                entity_name = entity_name.strip()
                if entity_name:
                    weighted_score = hyperedge_score
                    chunk_count = len(entity_to_chunk_ids.get(entity_name, set()))
                    if chunk_count > 0:
                        weighted_score /= chunk_count
                    entity_hyperedge_scores[entity_name] += weighted_score
                    entity_hyperedge_counts[entity_name] += 1
        # Noisy-OR: entities shared across more hyperedges get higher scores
        # while remaining bounded in [0, 1].  Each hyperedge is an independent
        # "observation"; the probability of NOT being relevant shrinks with count.
        #   score_final = 1 - (1 - avg) ** (1 + log(count))
        for entity in entity_hyperedge_scores:
            count = entity_hyperedge_counts[entity]
            if count > 0:
                avg_score = entity_hyperedge_scores[entity] / count
                new_val = 1.0 - (1.0 - avg_score) ** (1 + math.log(count))
                prev_val = query_param._entity_sims.get(entity, 0.0)
                query_param._entity_sims[entity] = max(new_val, prev_val)
    
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    relations_section_list = [
        ["id", "hyperedge", "related_entities"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["hyperedge"],
                e['related_nodes']
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN")
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(edge["hyperedge"]) for edge in edge_datas]
    )
    
    entity_names = []
    seen = set()

    for node_data in node_datas:
        for e in node_data:
            if e[1] not in seen:
                entity_names.append(e[1])
                seen.add(e[1])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
        if n is not None and n.get("role") == "entity"
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    result_text_units: list[TextChunkSchema] = [
        cast(TextChunkSchema, t["data"]) for t in truncated_text_units
    ]

    return result_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources

def _truncate_text_units(
    items: list,
    query_param: QueryParam,
    content_getter: Callable,
    score_key: str | None = None,
):
    """
    Generic truncation for text unit candidates.

    items: list of arbitrary objects (dicts)
    content_getter: function(item)-> str (raw text)
    score_key: if provided and mode == topk, items will be sorted descending by this key
    """
    if not items:
        return items
    mode = getattr(query_param, "text_unit_truncate_mode", "tokens")
    if mode == "topk":
        # Sort by score if available
        if score_key and all(score_key in it for it in items):
            items = sorted(items, key=lambda x: x[score_key], reverse=True)
        # Keep first top_k_text_units
        k = getattr(query_param, "top_k_text_units", 20)
        return items[:k]
    # Default: token-based truncation
    truncated = truncate_list_by_token_size(
        items,
        key=lambda x: content_getter(x),
        max_token_size=query_param.max_token_for_text_unit,
    )
    return truncated

async def build_graph_with_chunks(knowledge_graph_inst: BaseGraphStorage):
    """
    Build chunk nodes and connect them to entities and hyperedges based on source_ids.
    If include_hyperedge is False, remove hyperedge nodes from the graph.

    Args:
        graph_inst: The graph storage instance to build chunks for
        include_hyperedge: If True, create edges from hyperedges to chunks and keep them.
                          If False, remove hyperedge nodes from the graph.

    Returns:
        The modified graph instance with chunk nodes added
    """
    SEP_TOKEN = "<SEP>"
    nodes_data_list = await knowledge_graph_inst.get_all_nodes()

    for node_id, node_data in nodes_data_list:
        node_role = node_data.get("role", "")

        source_ids = node_data.get("source_id", "")
        if not source_ids or source_ids == "UNKNOWN":
            continue
        chunk_ids = [cid.strip() for cid in source_ids.split(SEP_TOKEN) if cid.strip()]
        for chunk_id in chunk_ids:
            if not await knowledge_graph_inst.has_node(chunk_id):
                await knowledge_graph_inst.upsert_node(chunk_id, {"role": "chunk"})
            await knowledge_graph_inst.upsert_edge(node_id, chunk_id, edge_data={})

    return knowledge_graph_inst

async def dense_chunk_retrieval(query_text, chunks_vdb, top_k):
    """
    Retrieve chunk IDs and similarity scores using dense retrieval, then min-max normalize.
    Returns: (sorted_chunk_ids, normalized_scores, chunk_sims_dict)
    """
    chunk_results = await chunks_vdb.query(query_text, top_k=top_k)
    chunk_ids = []
    chunk_scores = []
    for result in chunk_results:
        chunk_id = result.get("id")
        similarity = float(result.get("distance", 0.0))
        chunk_ids.append(chunk_id)
        chunk_scores.append(similarity)
    
    if not chunk_ids:
        logging.info("No chunks found in dense retrieval.")

    normalized_scores = min_max_normalize(chunk_scores)
    sorted_indices = np.argsort(normalized_scores)[::-1]
    sorted_chunk_ids = [chunk_ids[i] for i in sorted_indices[:top_k]]
    sorted_chunk_scores = normalized_scores[sorted_indices[:top_k]]
    return sorted_chunk_ids, sorted_chunk_scores

async def _ppr_rank_text_units(
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    chunks_vdb: BaseVectorStorage,
    alpha: float = 0.5, #default 0.4 best 0.3
    passage_node_weight: float = 0.5, # best 0.3, but with maud best 0.9
    top_ent_h: bool = True,
    top_n_ent: int = 5, # seems est 5, but 10 best for maud
    top_h_ent: int = 10,
    top_k_chunks: int = 100,
):
    """
    Build a small subgraph (entities + hyperedges + chunk ids) and run personalized PageRank.
    Personalization vector derives from embedding similarity stored in query_param._entity_sims / _hyperedge_sims.
    """
    logging.info("Running PPR ranking for text units...")
    local_kg_inst = copy.deepcopy(knowledge_graph_inst)
    local_kg_inst = await build_graph_with_chunks(local_kg_inst)

    entity_sims = getattr(query_param, "_entity_sims", {})
    hyper_sims = getattr(query_param, "_hyperedge_sims", {})
    entity_names_from_sims = list(entity_sims.keys())
    hyperedge_names_from_sims = list(hyper_sims.keys())
    
    nodes = await local_kg_inst.get_all_nodes()
    edges = await local_kg_inst.get_all_edges()

    logging.debug(f"[PPR] Graph nodes={len(nodes)}")

    # Map node names to indices
    idx_map = {n: i for i, n in enumerate(nodes)}
    pers = np.zeros(len(nodes))

    top_entity_names = set()
    top_hyperedge_names = set()
    if top_ent_h:
        top_entity_names = sorted(entity_names_from_sims, key=lambda n: entity_sims.get(n, 0.0), reverse=True)[:top_n_ent]
        top_hyperedge_names = sorted(hyperedge_names_from_sims, key=lambda h: hyper_sims.get(h, 0.0), reverse=True)[:top_h_ent]

    # --- Personalization mapping (node -> score) --------------------------------

    # Normalize entity and hyperedge scores to the same [0, 1] scale
    entity_vals = np.array([max(entity_sims.get(n, 0.0), 0.0) for n in entity_names_from_sims]) if entity_names_from_sims else np.array([])
    hyper_vals = np.array([max(hyper_sims.get(h, 0.0), 0.0) for h in hyperedge_names_from_sims]) if hyperedge_names_from_sims else np.array([])

    entity_vals_norm = min_max_normalize(entity_vals)
    hyper_vals_norm = min_max_normalize(hyper_vals)

    entity_sims_norm = {n: float(v) for n, v in zip(entity_names_from_sims, entity_vals_norm)} if len(entity_vals_norm) > 0 else {}
    hyper_sims_norm = {h: float(v) for h, v in zip(hyperedge_names_from_sims, hyper_vals_norm)} if len(hyper_vals_norm) > 0 else {}

    # Personalization for entities and hyperedges (using normalized scores)
    for n in entity_names_from_sims:
        if n in idx_map:
            pers[idx_map[n]] = entity_sims_norm.get(n, 0.0) if (not top_ent_h or n in top_entity_names) else 0.0
        else:
            logging.warning(f"[PPR] Entity '{n}' not found in graph nodes.")

    for h in hyperedge_names_from_sims:
                if h in idx_map:
                    pers[idx_map[h]] = hyper_sims_norm.get(h, 0.0) if (not top_ent_h or h in top_hyperedge_names) else 0.0
                else:
                    logging.warning(f"[PPR] Hyperedge '{h}' not found in graph nodes.")

    # --- Add chunk weights based on embedding similarity ---
    query_text = getattr(query_param, "query", "")
    sorted_chunk_ids = []
    normalized_scores = []
    if passage_node_weight > 0 and query_text:
        sorted_chunk_ids, normalized_scores = await dense_chunk_retrieval(
            query_text, chunks_vdb, top_k=top_k_chunks
        )

        # Return empty string if no relevant chunks found
        if normalized_scores is not None and len(normalized_scores) > 0:
            logging.info(f"[PPR] Applied passage weights to {len(sorted_chunk_ids)} chunks (min={min(normalized_scores):.4f}, max={max(normalized_scores):.4f}).")
        else:
            return ""
        # Attribute weights to chunk nodes
        for c_id, norm_score in zip(sorted_chunk_ids, normalized_scores):
                if c_id in idx_map:
                    pers[idx_map[c_id]] = norm_score * passage_node_weight
                else:
                    logging.warning(f"[PPR] Chunk '{c_id}' not found in graph nodes.")

    chunk_rows = []
    pers = np.where(np.isnan(pers) | (pers < 0), 0, pers)
    
    # Build igraph
    g = ig.Graph(directed=False)
    g.add_vertices(len(nodes))
    g.add_edges([(idx_map[src], idx_map[tgt]) for src, tgt, *_ in edges]) # type: ignore
    g.es["weight"] = [1.0] * len(g.es)
    
    try:
        rank = g.personalized_pagerank(
            vertices=range(len(nodes)),
            damping=alpha,
            directed=False,
            weights="weight",
            reset=pers,
            implementation="prpack"
        )
    except Exception as e:
        logging.warning(f"[PPR] igraph PRPACK failed ({e}), returning empty.")
        return ""    
    
    # Extract chunk node scores
    chunk_scores = []
    for c_id in nodes:
        if c_id in idx_map:
            node_data = await local_kg_inst.get_node(c_id)
            if node_data is not None and node_data.get("role") == "chunk":
                chunk_scores.append((c_id, rank[idx_map[c_id]]))
    
    if not chunk_scores:
        return ""

    # Sort chunks
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Fetch chunk contents
    for c_id, score in chunk_scores:
        row = await text_chunks_db.get_by_id(c_id)
        if not row or "content" not in row:
            continue
        chunk_rows.append({"id": c_id, "score": score, "content": row["content"]})


    # Truncate by configured mode (top-k or tokens budget)
    chunk_rows = _truncate_text_units(
        chunk_rows,
        query_param,
        content_getter=lambda x: x["content"],
        score_key="score",  # PPR provides score
    )

    # Build CSV
    csv_rows = [["id", "score", "content"]]
    for i, r in enumerate(chunk_rows):
        csv_rows.append([i, f"{r['score']:.4f}", r["content"]])
    return list_of_list_to_csv(csv_rows)