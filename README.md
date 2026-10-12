# RAG Assignment: Baseline vs Re-Ranking

Dataset: NITI Aayog National Strategy for Artificial Intelligence (114 pages, 362 chunks)
LLM: models/gemini-3.8-flash for final answers (lite model used temporarily for quota)
Bi-Encoder: all-MiniLM-L6-v2 with FAISS
Cross-Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2

## 1. Architecture Comparison

Bi-Encoder reads query and chunk separately. It makes one vector for query and one vector for each chunk. Then it compares with cosine similarity. In this project we used all-MiniLM-L6-v2 and FAISS IndexFlatIP. Vectors are made once and stored. Search is fast, 8 ms for 362 chunks. Problem is it does not see query and chunk together. So it fails on synonyms like COREs vs research centres and on words like not and without. Scores stay close, 0.55 to 0.58 for top 10.

Cross-Encoder reads query and chunk together as one pair. It uses joint attention across all tokens and gives direct relevance score. In this project we used ms-marco-MiniLM-L-6-v2 on top 10 pairs only. It is slow, 38 ms extra for 10 pairs, because it cannot precompute. But it understands context. It moved chunks from rank 8 to rank 1 and fixed Not found to Found.

In short, Bi-Encoder is fast retriever for large set. Cross-Encoder is slow reader for small set. So we use retrieve then re-rank. Top 10 first with Bi-Encoder, then re-rank to top 3 with Cross-Encoder.

## 2. Qualitative and Rank Shift Analysis

Query 1 - COREs funding amount per centre. Keyword trap with COREs term.
Rank changes: p55_172 old 4 new 1, p53_165 old 8 new 2.
Answer 1: Not found. Baseline top 3 missed gold chunk at rank 4 and 8, so LLM said not found and omitted essential clause on INR 50 to 100 crore.
Answer 2: Found. Gave INR 50 crore to 100 crore per CORE. Re-rank promoted gold chunk to top 1, so answer restored. No hallucination, exact match to page 55.

Query 2 - Why can AI adoption not succeed without data ecosystems and skilled workforce. Negation with not and without.
Rank changes: p7_15 old 8 new 1, p109_347 old 5 new 3.
Answer 1: Not found. Baseline blurred negative context and missed workforce chunk outside top 3.
Answer 2: Found. Gave data entry barrier plus lack of talent with 4 percent professionals on deep learning and PhD gap plus reskilling need. Restored by promotion from 8 to 1.

Query 3 - What barriers need to be addressed to achieve the goals of AI for All. General vs specific.
Rank changes: p45_137 old 4 new 2, p72_233 old 7 new 3.
Answer 1: Gave 8 points long list with duplicates. High cost repeated twice, privacy repeated twice. Noisy because baseline mixed general chunks.
Answer 2: Gave clean 4 points. Lack expertise, absence data ecosystems, high cost low awareness, privacy anonymisation. Precise match to page 6 gold chunk. No omission, no hallucination.

## 3. Latency Profiling

Measured on Kaggle GPU T4 x2, average of 5 runs, query COREs funding amount per centre, 362 chunks.

Stage: Baseline search Top 10 with FAISS. Time: 0.0081 sec. Note: fast because vectors precomputed.

Stage: Cross-Encoder re-rank for 10 pairs. Time: 0.0382 sec. Note: extra cost, about 5 times baseline.

Stage: Pipeline 1 total with LLM. Time: 0.377 sec. Note: includes Top 10 plus Top 3 plus Gemini lite call.

Stage: Pipeline 2 total with LLM. Time: 0.467 sec. Note: includes Top 10 plus re-rank plus Gemini lite call.

Extra cost for re-ranking with LLM: 0.09 sec. Retrieval without LLM goes from 0.0081 to 0.0463 sec.

Meaning: re-rank adds small extra time but gives better answers. LLM time dominates both pipelines. For 362 chunks this is fine. For lakhs of chunks, full Cross-Encoder scan would be too slow, so Top 10 filter is needed.

## 4. Scaling Question

We do not run Cross-Encoder on full collection because it is too slow and cannot precompute.

Bi-Encoder makes chunk vectors once and stores them. For a new query we make one query vector and do fast cosine in FAISS. Time grows slowly with size. For 362 chunks it is 8 ms. For lakhs of chunks it is still fast with index.

Cross-Encoder must read query plus each chunk together with joint attention. It cannot precompute. For each query it must run the full transformer for every pair. In our test 10 pairs took 38 ms. For 362 chunks it would take about 36 times more, around 1.4 sec per query without LLM. For 1 lakh chunks it would be hours. Too costly for live search.

So we use two stages. First Bi-Encoder filters to Top 10 fast. Then Cross-Encoder reads only 10 pairs carefully. This gives speed plus accuracy. Total retrieval goes from 8 ms to 46 ms only, but answer quality jumps from Not found to Found.

## How to run

1. Upload NITI_AI_Strategy.pdf as Kaggle dataset
2. Run notebook cells in order: chunking, embeddings, FAISS, search, rerank, answers
3. Same prompt used for Answer 1 and Answer 2, only chunks differ
