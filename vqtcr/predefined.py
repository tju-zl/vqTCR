# ESM tokenizer vocabulary, tips: '-' can't be add to adata.uns
TCR_VOCAB = {
  "<cls>": 0,
  "<pad>": 1,
  "<eos>": 2,
  "<unk>": 3,
  "L": 4,
  "A": 5,
  "G": 6,
  "V": 7,
  "S": 8,
  "E": 9,
  "R": 10,
  "T": 11,
  "I": 12,
  "D": 13,
  "P": 14,
  "K": 15,
  "Q": 16,
  "N": 17,
  "F": 18,
  "Y": 19,
  "M": 20,
  "H": 21,
  "W": 22,
  "C": 23,
  "X": 24,
  "B": 25,
  "U": 26,
  "Z": 27,
  "O": 28,
  ".": 29,
  "-": 30,
  "<null_1>": 31,
  "<mask>": 32
}

# You can validate the esm vocab using esm2 tokenizer.
# https://github.com/facebookresearch/esm
# https://github.com/huggingface/transformers
esm_type = 'facebook/esm2_t6_8M_UR50D'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(esm_type)
tokenizer.get_vocab()

# 10X data high count antigens (get from ./pre_data/tutorial_prep_10.ipynb)
HIGH_COUNT_ANTIGENS = ['A0201_ELAGIGILTV_MART-1_Cancer_binder',
					   'A0201_GILGFVFTL_Flu-MP_Influenza_binder',
					   'A0201_GLCTLVAML_BMLF1_EBV_binder',
					   'A0301_KLGGALQAK_IE-1_CMV_binder',
					   'A0301_RLRAEAQVK_EMNA-3A_EBV_binder',
					   'A1101_IVTDFSVIK_EBNA-3B_EBV_binder',
					   'A1101_AVFDRKSDAK_EBNA-3B_EBV_binder',
					   'B0801_RAKFKQLL_BZLF1_EBV_binder']

# mvTCR vocab (get from mvTCR)
aa_to_id = {'+': 21,
			'<': 22,
			'>': 23,
			'A': 1,
			'C': 2,
			'D': 3,
			'E': 4,
			'F': 5,
			'G': 6,
			'H': 7,
			'I': 8,
			'K': 9,
			'L': 10,
			'M': 11,
			'N': 12,
			'P': 13,
			'Q': 14,
			'R': 15,
			'S': 16,
			'T': 17,
			'V': 18,
			'W': 19,
			'Y': 20,
			'_': 0}