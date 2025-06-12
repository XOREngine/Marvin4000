// Marvin4000 - Real-time Audio Transcription & Translation
// © 2025 XOREngine (WallyByte)
// https://github.com/XOREngine/marvin4000

const claude_references = [
  "...",
  "...",
  "...",
];

const marvin_hypothesis = [
  "...",
  "...",
  "...",
];

function calculate_bleu_standard(hypothesis, reference) {
  function tokenize(text) {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);
  }
  
  const tokens_hyp = tokenize(hypothesis);
  const tokens_ref = tokenize(reference);
  
  console.log(`Tokens Hyp: [${tokens_hyp.join(', ')}]`);
  console.log(`Tokens Ref: [${tokens_ref.join(', ')}]`);
  console.log(`Length Hyp: ${tokens_hyp.length}, Length Ref: ${tokens_ref.length}`);
  
  if (tokens_hyp.length === 0) {
    return { bleu1: 0, bleu4: 0, corpus_bleu: 0, bp: 0 };
  }
  
  // N-gram precisions (1 to 4)
  const precisions = [];
  
  for (let n = 1; n <= 4; n++) {
    console.log(`\n--- Calculating ${n}-grams ---`);
    
    const ngrams_hyp = [];
    const ngrams_ref = [];
    
    for (let i = 0; i <= tokens_hyp.length - n; i++) {
      ngrams_hyp.push(tokens_hyp.slice(i, i + n).join(' '));
    }
    
    for (let i = 0; i <= tokens_ref.length - n; i++) {
      ngrams_ref.push(tokens_ref.slice(i, i + n).join(' '));
    }
    
    console.log(`${n}-grams Hyp: [${ngrams_hyp.join(', ')}]`);
    console.log(`${n}-grams Ref: [${ngrams_ref.join(', ')}]`);
    
    if (ngrams_hyp.length === 0) {
      precisions.push(0);
      continue;
    }
    
    // Count with clipping
    const ref_counts = {};
    ngrams_ref.forEach(ngram => {
      ref_counts[ngram] = (ref_counts[ngram] || 0) + 1;
    });
    
    const hyp_counts = {};
    ngrams_hyp.forEach(ngram => {
      hyp_counts[ngram] = (hyp_counts[ngram] || 0) + 1;
    });
    
    console.log(`Ref counts: ${JSON.stringify(ref_counts)}`);
    console.log(`Hyp counts: ${JSON.stringify(hyp_counts)}`);
    
    let clipped_counts = 0;
    for (const ngram in hyp_counts) {
      if (ref_counts[ngram]) {
        const clipped = Math.min(hyp_counts[ngram], ref_counts[ngram]);
        clipped_counts += clipped;
        console.log(`"${ngram}": min(${hyp_counts[ngram]}, ${ref_counts[ngram]}) = ${clipped}`);
      }
    }
    
    const precision = clipped_counts / ngrams_hyp.length;
    console.log(`Precision ${n}: ${clipped_counts}/${ngrams_hyp.length} = ${precision}`);
    precisions.push(precision);
  }
  
  // Brevity penalty
  const bp = tokens_ref.length === 0 ? 0 : 
    Math.min(1.0, Math.exp(1 - tokens_ref.length / tokens_hyp.length));
  
  console.log(`BP: min(1, exp(1 - ${tokens_ref.length}/${tokens_hyp.length})) = ${bp}`);
  
  // BLEU score
  console.log(`Precisions: [${precisions.map(p => p.toFixed(4)).join(', ')}]`);
  
  if (precisions.includes(0)) {
    console.log("BLEU = 0 (some precision is 0)");
    return {
      bleu1: precisions[0],
      bleu4: precisions[3],
      corpus_bleu: 0,
      bp: bp
    };
  }
  
  const geometric_mean = Math.pow(
    precisions[0] * precisions[1] * precisions[2] * precisions[3], 0.25
  );
  const corpus_bleu = bp * geometric_mean;
  
  console.log(`Geometric mean: (${precisions.map(p => p.toFixed(4)).join(' * ')})^0.25 = ${geometric_mean}`);
  console.log(`Corpus BLEU: ${bp} * ${geometric_mean} = ${corpus_bleu}`);
  
  return {
    bleu1: precisions[0],
    bleu4: precisions[3],
    corpus_bleu: corpus_bleu,
    bp: bp
  };
}

function calculate_corpus_bleu(hypothesis, references) {
  console.log("=== CALCULATING CORPUS BLEU ===");
  
  let total_tokens_hyp = 0;
  let total_tokens_ref = 0;
  const total_matches = [0, 0, 0, 0]; // for n=1,2,3,4
  const total_ngrams = [0, 0, 0, 0];
  
  function tokenize(text) {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);
  }
  
  // Aggregate statistics from all segments
  for (let i = 0; i < hypothesis.length; i++) {
    console.log(`\n--- Processing segment ${i+1} ---`);
    
    const tokens_hyp = tokenize(hypothesis[i]);
    const tokens_ref = tokenize(references[i]);
    
    total_tokens_hyp += tokens_hyp.length;
    total_tokens_ref += tokens_ref.length;
    
    console.log(`Segment ${i+1}: Hyp=${tokens_hyp.length} tokens, Ref=${tokens_ref.length} tokens`);
    
    // For each n-gram
    for (let n = 1; n <= 4; n++) {
      const ngrams_hyp = [];
      const ngrams_ref = [];
      
      for (let j = 0; j <= tokens_hyp.length - n; j++) {
        ngrams_hyp.push(tokens_hyp.slice(j, j + n).join(' '));
      }
      
      for (let j = 0; j <= tokens_ref.length - n; j++) {
        ngrams_ref.push(tokens_ref.slice(j, j + n).join(' '));
      }
      
      total_ngrams[n-1] += ngrams_hyp.length;
      
      // Count matches with clipping
      const ref_counts = {};
      ngrams_ref.forEach(ngram => {
        ref_counts[ngram] = (ref_counts[ngram] || 0) + 1;
      });
      
      const hyp_counts = {};
      ngrams_hyp.forEach(ngram => {
        hyp_counts[ngram] = (hyp_counts[ngram] || 0) + 1;
      });
      
      let segment_matches = 0;
      for (const ngram in hyp_counts) {
        if (ref_counts[ngram]) {
          segment_matches += Math.min(hyp_counts[ngram], ref_counts[ngram]);
        }
      }
      
      total_matches[n-1] += segment_matches;
      
      if (n === 1) {
        console.log(`  ${n}-grams: ${segment_matches}/${ngrams_hyp.length} matches`);
      }
    }
  }
  
  console.log("\n=== CORPUS TOTALS ===");
  console.log(`Total tokens Hyp: ${total_tokens_hyp}`);
  console.log(`Total tokens Ref: ${total_tokens_ref}`);
  
  // Calculate corpus-level precisions
  const precisions = total_ngrams.map((total, i) => {
    const precision = total > 0 ? total_matches[i] / total : 0;
    console.log(`BLEU-${i+1}: ${total_matches[i]}/${total} = ${precision.toFixed(4)}`);
    return precision;
  });
  
  // Corpus-level brevity penalty
  const bp = total_tokens_hyp === 0 ? 0 : 
    Math.min(1.0, Math.exp(1 - total_tokens_ref / total_tokens_hyp));
  
  console.log(`BP corpus: min(1, exp(1 - ${total_tokens_ref}/${total_tokens_hyp})) = ${bp.toFixed(4)}`);
  
  // Corpus BLEU
  const corpus_bleu = precisions.includes(0) ? 0 : 
    bp * Math.pow(precisions[0] * precisions[1] * precisions[2] * precisions[3], 0.25);
  
  console.log(`Corpus BLEU: ${bp.toFixed(4)} * (${precisions.map(p => p.toFixed(4)).join(' * ')})^0.25 = ${corpus_bleu.toFixed(4)}`);
  
  return {
    bleu1_corpus: precisions[0],
    bleu4_corpus: precisions[3],
    corpus_bleu: corpus_bleu,
    bp: bp,
    total_tokens_hyp: total_tokens_hyp,
    total_tokens_ref: total_tokens_ref
  };
}

console.log("BLEU ANALYSIS MARVIN4000");
console.log("====================================================");

// Calculate by segments (first 3 for debug)
console.log("\nSEGMENT ANALYSIS (first 3):");
for (let i = 0; i < 3; i++) {
  console.log(`\nSEGMENT ${i+1}:`);
  console.log(`Hyp: "${marvin_hypothesis[i]}"`);
  console.log(`Ref: "${claude_references[i]}"`);
  
  const result = calculate_bleu_standard(marvin_hypothesis[i], claude_references[i]);
  
  console.log(`RESULT: BLEU-1=${(result.bleu1*100).toFixed(1)}%, BLEU-4=${(result.bleu4*100).toFixed(1)}%, Corpus=${(result.corpus_bleu*100).toFixed(1)}%`);
  console.log("=".repeat(50));
}

const corpus_result = calculate_corpus_bleu(marvin_hypothesis, claude_references);

console.log("\nFINAL RESULTS MARVIN4000:");
console.log(`BLEU-1 Corpus: ${(corpus_result.bleu1_corpus * 100).toFixed(1)}%`);
console.log(`BLEU-4 Corpus: ${(corpus_result.bleu4_corpus * 100).toFixed(1)}%`);
console.log(`Corpus BLEU: ${(corpus_result.corpus_bleu * 100).toFixed(1)}%`);
console.log(`BP: ${corpus_result.bp.toFixed(3)}`);
console.log(`Total segments: ${marvin_hypothesis.length}`);