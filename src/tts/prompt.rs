use crate::assets_manager::Assets;
use crate::utils::tokenizer::Tokenizer;

// Constants from PROTOCOL
pub const PAD: usize = 2148;
pub const BOS: usize = 2149;
pub const EOS: usize = 2150;
pub const BOS_TOKEN: usize = 151672;
pub const EOS_TOKEN: usize = 151673;
pub const THINK: usize = 2154;
pub const NOTHINK: usize = 2155;
pub const THINK_BOS: usize = 2156;
pub const THINK_EOS: usize = 2157;

// Magic token derived from Python implementation (assets.text_table[151671])
pub const TEXT_AUDIO_MARKER: usize = 151671;

pub struct PromptData {
    pub embd: Vec<Vec<f32>>, // Flattened or vector of vectors? Python returns (1, seq, 2048). Here Vec<Vec<f32>> usually [seq][2048]
    pub text_ids: Vec<u32>,
    pub spk_emb: Vec<f32>,
}

pub struct PromptBuilder;

impl PromptBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn build_clone_prompt(
        text: &str,
        tokenizer: &Tokenizer,
        assets: &Assets,
        ref_codes: &[i32],
        ref_text_ids: &[u32],
        spk_emb: &[f32],
        lang_id: usize,
        instruct: Option<&str>,
    ) -> PromptData {
        let mut mid_embeds = Vec::new();

        // 1. Inject Identity Overlay (Text): BOS_TOKEN -> ID -> EOS_TOKEN
        let mut ref_ids_full = vec![BOS_TOKEN as u32];
        ref_ids_full.extend_from_slice(ref_text_ids);
        ref_ids_full.push(EOS_TOKEN as u32);

        // Pre-fetch PAD from elem 0 (assuming codec 0)
        // Python: assets.emb_tables[0][p["PAD"]]
        let pad_emb = assets.get_codec_embedding(0, PAD as i32);

        for &tid in &ref_ids_full {
            let t_emb = assets.get_text_embedding(tid as usize);
            // sum = t_emb + pad_emb
            let summed: Vec<f32> = t_emb
                .iter()
                .zip(pad_emb.iter())
                .map(|(a, b)| a + b)
                .collect();
            mid_embeds.push(summed);
        }

        // 2. Inject Audio Codes (Codec BOS -> Codes -> PAD)
        // Python: Codec BOS (text[151671] + emb[0][2160]) ? Wait, check python code carefully.
        // Python: assets.text_table[151671] + assets.emb_tables[0][2160] # Codec BOS?
        // Wait, 2160 is not in PROTOCOL constants shown, but likely is in user's logic.
        // Let's assume 2160 is a magic constant for Codec Start?
        // Actually, let's look at prompt_builder.py again.
        // "assets.text_table[151671] + assets.emb_tables[0][2160]"
        let marker_emb = assets.get_text_embedding(TEXT_AUDIO_MARKER);
        let codec_bos_emb = assets.get_codec_embedding(0, 2160); // Hardcoded 2160
        let start_sum: Vec<f32> = marker_emb
            .iter()
            .zip(codec_bos_emb.iter())
            .map(|(a, b)| a + b)
            .collect();
        mid_embeds.push(start_sum);

        // Codes Loop
        // ref_codes should be flattened, but we need steps.
        // We assume ref_codes is [Steps * 16]
        let n_steps = ref_codes.len() / 16;
        for step in 0..n_steps {
            let mut summed_c = vec![0.0; 2048];
            for q in 0..16 {
                let c = ref_codes[step * 16 + q];
                let emb = assets.get_codec_embedding(q, c);
                for i in 0..2048 {
                    summed_c[i] += emb[i];
                }
            }
            // Add marker
            let final_vec: Vec<f32> = marker_emb
                .iter()
                .zip(summed_c.iter())
                .map(|(a, b)| a + b)
                .collect();
            mid_embeds.push(final_vec);
        }

        // Add Pad at end of audio
        // Python: assets.text_table[151671] + assets.emb_tables[0][p["PAD"]]
        let pad_0 = assets.get_codec_embedding(0, PAD as i32);
        let end_sum: Vec<f32> = marker_emb
            .iter()
            .zip(pad_0.iter())
            .map(|(a, b)| a + b)
            .collect();
        mid_embeds.push(end_sum);

        Self::build_core(
            text,
            tokenizer,
            assets,
            Some(lang_id),
            None,
            Some(spk_emb),
            instruct,
            Some(mid_embeds),
        )
    }

    pub fn build_custom_prompt(
        text: &str,
        tokenizer: &Tokenizer,
        assets: &Assets,
        spk_id: usize,
        lang_id: usize,
        instruct: Option<&str>,
    ) -> PromptData {
        Self::build_core(
            text,
            tokenizer,
            assets,
            Some(lang_id),
            Some(spk_id),
            None,
            instruct, // Pass instruction here
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_core(
        text: &str,
        tokenizer: &Tokenizer,
        assets: &Assets,
        lang_id: Option<usize>,
        spk_id: Option<usize>,
        spk_emb: Option<&[f32]>,
        instruct: Option<&str>,
        mid_embeds: Option<Vec<Vec<f32>>>,
    ) -> PromptData {
        let mut embeds = Vec::new();

        // 1. Instruct Block (User)
        if let Some(ins) = instruct {
            // <|im_start|>user\n
            let prefix = vec![151644, 872, 198];
            for id in prefix {
                embeds.push(assets.get_text_embedding(id));
            }
            let content_ids = tokenizer.encode(ins);
            for id in content_ids {
                embeds.push(assets.get_text_embedding(id as usize));
            }
            // <|im_end|>\n
            let suffix = vec![151645, 198];
            for id in suffix {
                embeds.push(assets.get_text_embedding(id));
            }
        }

        // 2. Role Block (Assistant)
        // <|im_start|>assistant\n -> [151644, 77091, 198]
        for id in [151644, 77091, 198] {
            embeds.push(assets.get_text_embedding(id));
        }

        let marker_emb = assets.get_text_embedding(TEXT_AUDIO_MARKER);

        // 3. Control Block
        if let Some(lid) = lang_id {
            // THINK, THINK_BOS, lang_id, THINK_EOS
            let ids = [THINK, THINK_BOS, lid, THINK_EOS];
            for &id in &ids {
                let e = assets.get_codec_embedding(0, id as i32);
                let sum: Vec<f32> = marker_emb
                    .iter()
                    .zip(e.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                embeds.push(sum);
            }
        } else {
            // NOTHINK, THINK_BOS, THINK_EOS
            let ids = [NOTHINK, THINK_BOS, THINK_EOS];
            for &id in &ids {
                let e = assets.get_codec_embedding(0, id as i32);
                let sum: Vec<f32> = marker_emb
                    .iter()
                    .zip(e.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                embeds.push(sum);
            }
        }

        // Speaker ID/Emb
        if let Some(sid) = spk_id {
            let e = assets.get_codec_embedding(0, sid as i32);
            let sum: Vec<f32> = marker_emb
                .iter()
                .zip(e.iter())
                .map(|(a, b)| a + b)
                .collect();
            embeds.push(sum);
        } else if let Some(se) = spk_emb {
            let sum: Vec<f32> = marker_emb
                .iter()
                .zip(se.iter())
                .map(|(a, b)| a + b)
                .collect();
            embeds.push(sum);
        }

        // 4. Mid Embeds
        if let Some(mids) = mid_embeds {
            embeds.extend(mids);
        }

        // 5. Task Text Block
        let ids = tokenizer.encode(text);
        // BOS_TOKEN + PAD
        let pad_0 = assets.get_codec_embedding(0, PAD as i32);
        let bos_token_emb = assets.get_text_embedding(BOS_TOKEN);
        let bos_sum: Vec<f32> = bos_token_emb
            .iter()
            .zip(pad_0.iter())
            .map(|(a, b)| a + b)
            .collect();
        embeds.push(bos_sum);

        for &id in &ids {
            let t_emb = assets.get_text_embedding(id as usize);
            let sum: Vec<f32> = t_emb.iter().zip(pad_0.iter()).map(|(a, b)| a + b).collect();
            embeds.push(sum);
        }

        // EOS_TOKEN + PAD
        let eos_token_emb = assets.get_text_embedding(EOS_TOKEN);
        let eos_sum: Vec<f32> = eos_token_emb
            .iter()
            .zip(pad_0.iter())
            .map(|(a, b)| a + b)
            .collect();
        embeds.push(eos_sum);

        // 6. Activation (BOS)
        // Python: assets.text_table[151671] + assets.emb_tables[0][p["BOS"]]
        let bos_emb = assets.get_codec_embedding(0, BOS as i32);
        let act_sum: Vec<f32> = marker_emb
            .iter()
            .zip(bos_emb.iter())
            .map(|(a, b)| a + b)
            .collect();
        embeds.push(act_sum);

        let result_spk_emb = if let Some(se) = spk_emb {
            se.to_vec()
        } else {
            vec![0.0; 2048]
        };

        PromptData {
            embd: embeds,
            text_ids: ids.into_iter().collect(),
            spk_emb: result_spk_emb,
        }
    }
}
