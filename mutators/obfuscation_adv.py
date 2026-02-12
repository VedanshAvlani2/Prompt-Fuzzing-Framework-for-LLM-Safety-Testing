class ObfuscationAdvancedMutator:
    """Advanced obfuscation techniques from teammate's framework"""

    def mutate(self, prompt: str, params=None):
        pid = str(abs(hash(prompt)) % 10000)

        # Leetspeak version
        leet = (
            prompt.replace('a', '4')
                  .replace('e', '3')
                  .replace('i', '1')
                  .replace('o', '0')
        )

        return [
            {
                "text": f"Decode this request and respond: {prompt}",
                "mutation_type": "encoding_obfuscation",
                "prompt_id": f"obf_encode_{pid}",
                "params": {"method": "decode_request"}
            },
            {
                "text": leet,
                "mutation_type": "leetspeak_obfuscation",
                "prompt_id": f"obf_leet_{pid}",
                "params": {"method": "character_substitution"}
            }
        ]
