use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use fancy_regex::Regex;
use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;

type Rank = u32;

const O200K_RANKS: &str = include_str!("../../config/o200k_base.tiktoken");

const O200K_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+(?!\S)",
    "|",
    r"\s+",
);

// Ported from openai/tiktoken 0.13.0 CoreBPE encode_ordinary and byte-pair merge
// routines at commit 08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa (MIT).
// The embedded o200k_base ranks have SHA-256
// 446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d.
// Last synchronized with upstream: 2026-07-14.
struct O200kTokenizer {
    ranks: HashMap<Vec<u8>, Rank>,
    regex: Regex,
}

impl O200kTokenizer {
    fn new() -> Self {
        let mut ranks = HashMap::with_capacity(199_998);
        for (line_index, line) in O200K_RANKS.lines().enumerate() {
            let (encoded, rank) = line.split_once(' ').unwrap_or_else(|| {
                panic!(
                    "invalid embedded o200k_base rank at line {}",
                    line_index + 1
                )
            });
            let token = BASE64_STANDARD.decode(encoded).unwrap_or_else(|error| {
                panic!(
                    "invalid base64 in embedded o200k_base rank at line {}: {error}",
                    line_index + 1
                )
            });
            let rank = rank.parse::<Rank>().unwrap_or_else(|error| {
                panic!(
                    "invalid rank in embedded o200k_base data at line {}: {error}",
                    line_index + 1
                )
            });
            assert!(
                ranks.insert(token, rank).is_none(),
                "duplicate token in embedded o200k_base data at line {}",
                line_index + 1
            );
        }
        assert_eq!(
            ranks.len(),
            199_998,
            "embedded o200k_base data must contain 199,998 mergeable ranks"
        );
        let regex = Regex::new(O200K_PATTERN).expect("embedded o200k_base regex must compile");
        Self { ranks, regex }
    }

    fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let mut tokens = Vec::new();
        for match_result in self.regex.find_iter(text) {
            let piece = match_result
                .expect("o200k_base regex matching must not fail")
                .as_str()
                .as_bytes();
            if let Some(&rank) = self.ranks.get(piece) {
                tokens.push(rank);
            } else {
                tokens.extend(byte_pair_encode(piece, &self.ranks));
            }
        }
        tokens
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Merge {
    start: usize,
    rank: Rank,
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.start.cmp(&self.start))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct MergeState {
    prev: usize,
    end: usize,
    next_end: usize,
    next_rank: Rank,
    current_rank: Rank,
}

fn byte_pair_merge_large(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<Rank> {
    let mut state = Vec::with_capacity(piece.len());
    state.push(MergeState {
        prev: usize::MAX,
        end: 1,
        next_end: 2,
        next_rank: Rank::MAX,
        current_rank: Rank::MAX,
    });

    let mut heap = BinaryHeap::with_capacity(piece.len());
    for index in 0..piece.len() - 1 {
        if let Some(&rank) = ranks.get(&piece[index..index + 2]) {
            heap.push(Merge { start: index, rank });
            state[index].next_rank = rank;
        }
        state.push(MergeState {
            prev: index,
            end: index + 2,
            next_end: index + 3,
            next_rank: Rank::MAX,
            current_rank: Rank::MAX,
        });
    }

    let potential_merge = |state: &mut Vec<MergeState>,
                           heap: &mut BinaryHeap<Merge>,
                           start: usize,
                           next_end: usize| {
        state[start].next_end = next_end;
        state[start].next_rank = Rank::MAX;
        if next_end <= piece.len() {
            if let Some(&rank) = ranks.get(&piece[start..next_end]) {
                heap.push(Merge { start, rank });
                state[start].next_rank = rank;
            }
        }
    };

    while let Some(left) = heap.pop() {
        if left.rank == Rank::MAX {
            break;
        }
        if left.rank != state[left.start].next_rank {
            continue;
        }

        let left_start = left.start;
        let right_start = state[left_start].end;
        let right_end = state[left_start].next_end;
        debug_assert_eq!(right_end, state[right_start].end);
        let right_next_end = state[right_start].next_end;

        state[left_start].current_rank = state[left_start].next_rank;
        state[left_start].end = right_end;
        potential_merge(&mut state, &mut heap, left_start, right_next_end);
        if right_end < state.len() {
            state[right_end].prev = left_start;
        }
        if left_start > 0 {
            let previous_start = state[left_start].prev;
            potential_merge(&mut state, &mut heap, previous_start, right_end);
        }
        state[right_start].next_rank = Rank::MAX;
    }

    let mut tokens = Vec::new();
    let mut index = 0;
    while index < state.len() {
        if state[index].current_rank != Rank::MAX {
            tokens.push(state[index].current_rank);
        } else {
            tokens.push(ranks[&piece[index..state[index].end]]);
        }
        index = state[index].end;
    }
    tokens
}

fn byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);
    let mut minimum = (Rank::MAX, usize::MAX);
    for index in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[index..index + 2]).unwrap_or(&Rank::MAX);
        if rank < minimum.0 {
            minimum = (rank, index);
        }
        parts.push((index, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = |parts: &Vec<(usize, Rank)>, index: usize| {
        if index + 3 < parts.len() {
            *ranks
                .get(&piece[parts[index].0..parts[index + 3].0])
                .unwrap_or(&Rank::MAX)
        } else {
            Rank::MAX
        }
    };

    while minimum.0 != Rank::MAX {
        let index = minimum.1;
        if index > 0 {
            parts[index - 1].1 = get_rank(&parts, index - 1);
        }
        parts[index].1 = get_rank(&parts, index);
        parts.remove(index + 1);

        minimum = (Rank::MAX, usize::MAX);
        for (index, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < minimum.0 {
                minimum = (rank, index);
            }
        }
    }
    parts
}

fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    if piece.len() < 100 {
        return byte_pair_merge(ranks, piece)
            .windows(2)
            .map(|part| ranks[&piece[part[0].0..part[1].0]])
            .collect();
    }
    byte_pair_merge_large(ranks, piece)
}

fn tokenizer() -> &'static O200kTokenizer {
    static TOKENIZER: OnceLock<O200kTokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(O200kTokenizer::new)
}

pub fn encode_ordinary(text: &str) -> Vec<u32> {
    tokenizer().encode_ordinary(text)
}

pub fn count_ordinary(text: &str) -> usize {
    encode_ordinary(text).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::time::{Duration, Instant};

    #[derive(Deserialize)]
    struct ReferenceCase {
        text: String,
        tokens: Vec<u32>,
    }

    #[test]
    fn encode_ordinary_matches_official_o200k_reference_cases() {
        let cases: Vec<ReferenceCase> = serde_json::from_str(include_str!(
            "../../tests/fixtures/o200k_base_encode_ordinary.json"
        ))
        .unwrap();
        for case in cases {
            assert_eq!(encode_ordinary(&case.text), case.tokens, "{}", case.text);
        }
    }

    #[test]
    fn encode_ordinary_uses_large_piece_path_without_quadratic_runtime() {
        let text = "abcd".repeat(1_000);
        let _ = encode_ordinary("warmup");
        let started = Instant::now();
        let tokens = encode_ordinary(&text);
        let elapsed = started.elapsed();

        assert_eq!(tokens.len(), 1_000);
        assert!(
            elapsed < Duration::from_secs(2),
            "4,000-byte single-piece encode took {elapsed:?}"
        );
    }
}
