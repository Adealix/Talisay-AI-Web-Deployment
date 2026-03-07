/**
 * Bad Words Filter Utility
 * Filters profanity and inappropriate language from user-generated content.
 * Used in Forum posts/comments and Chatbot messages.
 */

// Comprehensive bad words list (common English profanity)
const BAD_WORDS = [
  // Common profanity
  'fuck', 'shit', 'ass', 'damn', 'hell', 'bitch', 'bastard', 'dick', 'cock',
  'pussy', 'cunt', 'whore', 'slut', 'fag', 'faggot', 'nigger', 'nigga',
  'retard', 'retarded',
  // Variations and common misspellings
  'f u c k', 'sh1t', 'b1tch', 'a$$', 'd1ck', 'fck', 'fcking', 'stfu',
  'wtf', 'lmfao', 'gtfo',
  'motherfucker', 'motherfucking', 'fucker', 'fucking', 'fucked', 'fucks',
  'shitty', 'shitting', 'bullshit', 'horseshit', 'dipshit',
  'asshole', 'arsehole', 'asswipe', 'dumbass', 'jackass', 'badass',
  'bitchy', 'bitches', 'bitching', 'sonofabitch',
  'dammit', 'goddamn', 'goddammit',
  'dickhead', 'dickwad',
  'piss', 'pissed', 'pissing',
  'crap', 'crappy',
  'wanker', 'tosser', 'twat', 'bollocks', 'bugger',
  'boob', 'boobs', 'tits', 'titties',
  'porn', 'porno', 'pornography',
  // Filipino profanity
  'putangina', 'putang ina', 'tangina', 'tang ina', 'puta', 'gago', 'gaga',
  'tanga', 'bobo', 'bwisit', 'leche', 'lintik', 'ulol', 'ungas',
  'tarantado', 'torpe', 'pakyu', 'punyeta', 'hudas', 'hayop ka',
  'amputa', 'kingina', 'kinginamo', 'inamo', 'inamu',
  'pesteng yawa', 'yawa', 'animal ka', 'kupal',
  'burat', 'tite', 'pepe', 'pekpek', 'kantot', 'jakol',
  'hindot', 'pokpok', 'malibog', 'libog',
];

// Build regex patterns for each bad word (word boundary matching)
const BAD_WORD_PATTERNS = BAD_WORDS.map(word => {
  // Escape special regex characters
  const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  // Create pattern that matches the word with optional letter substitutions
  return new RegExp(escaped, 'gi');
});

/**
 * Check if text contains bad words.
 * @param {string} text - Text to check
 * @returns {{ hasBadWords: boolean, detectedWords: string[] }}
 */
export function checkForBadWords(text) {
  if (!text || typeof text !== 'string') return { hasBadWords: false, detectedWords: [] };

  const lowerText = text.toLowerCase().trim();
  const detectedWords = [];

  for (const word of BAD_WORDS) {
    // Always use word-boundary matching to prevent false positives
    // e.g. "hell" should NOT match "hello", "ass" should NOT match "class"
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`\\b${escaped}\\b`, 'gi');
    if (pattern.test(lowerText)) {
      detectedWords.push(word);
    }
  }

  return {
    hasBadWords: detectedWords.length > 0,
    detectedWords: [...new Set(detectedWords)],
  };
}

/**
 * Censor bad words in text by replacing with asterisks.
 * @param {string} text - Text to censor
 * @returns {string} Censored text
 */
export function censorBadWords(text) {
  if (!text || typeof text !== 'string') return text;

  let censored = text;
  for (const word of BAD_WORDS) {
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    // Always use word boundaries to avoid clobbering innocent words
    const pattern = new RegExp(`\\b(${escaped})\\b`, 'gi');
    censored = censored.replace(pattern, (match) => {
      if (match.length <= 1) return '*';
      return match[0] + '*'.repeat(match.length - 2) + match[match.length - 1];
    });
  }
  return censored;
}

/**
 * Get a user-friendly warning message for detected bad words.
 * @returns {string}
 */
export function getBadWordWarning() {
  return 'Your message contains inappropriate language. Please keep the discussion respectful and professional.';
}
