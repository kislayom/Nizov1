package ai.nizo.memory.util;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class UtilsTest {

    // ---------------------------------------------------------------
    //  Vectors
    // ---------------------------------------------------------------
    @Nested
    class VectorsTest {

        @Test
        void cosineSameVectorReturnsOne() {
            float[] v = {1, 2, 3};
            assertEquals(1.0, Vectors.cosine(v, v), 1e-9);
        }

        @Test
        void cosineOppositeVectorsReturnsNegativeOne() {
            float[] a = {1, 2, 3};
            float[] b = {-1, -2, -3};
            assertEquals(-1.0, Vectors.cosine(a, b), 1e-9);
        }

        @Test
        void cosineOrthogonalVectorsReturnsZero() {
            float[] a = {1, 0, 0};
            float[] b = {0, 1, 0};
            assertEquals(0.0, Vectors.cosine(a, b), 1e-9);
        }

        @Test
        void cosineNullReturnsZero() {
            float[] v = {1, 2, 3};
            assertEquals(0.0, Vectors.cosine(null, v));
            assertEquals(0.0, Vectors.cosine(v, null));
            assertEquals(0.0, Vectors.cosine(null, null));
        }

        @Test
        void cosineZeroVectorReturnsZero() {
            float[] zero = {0, 0, 0};
            float[] v = {1, 2, 3};
            assertEquals(0.0, Vectors.cosine(zero, v));
            assertEquals(0.0, Vectors.cosine(v, zero));
            assertEquals(0.0, Vectors.cosine(zero, zero));
        }

        @Test
        void cosineDifferentLengthsReturnsZero() {
            float[] a = {1, 2};
            float[] b = {1, 2, 3};
            assertEquals(0.0, Vectors.cosine(a, b));
        }

        @Test
        void cosineRealistic768DimVectors() {
            Random rng = new Random(42);
            float[] a = randomVector(768, rng);
            float[] b = randomVector(768, rng);
            double sim = Vectors.cosine(a, a);
            assertEquals(1.0, sim, 1e-6, "self-similarity should be ~1.0");

            double cross = Vectors.cosine(a, b);
            assertTrue(cross > -1.0 && cross < 1.0,
                    "random vectors should have cosine in (-1, 1), got " + cross);
        }

        // -- toBytes / fromBytes round-trip --

        @Test
        void toBytesFromBytesRoundTrip() {
            float[] original = {1.5f, -3.14f, 0.0f, Float.MAX_VALUE, Float.MIN_VALUE};
            float[] restored = Vectors.fromBytes(Vectors.toBytes(original));
            assertArrayEquals(original, restored);
        }

        @Test
        void toBytesFromBytesEmptyArray() {
            float[] empty = {};
            float[] restored = Vectors.fromBytes(Vectors.toBytes(empty));
            assertEquals(0, restored.length);
        }

        @Test
        void toBytesFromBytesSingleElement() {
            float[] single = {42.0f};
            assertArrayEquals(single, Vectors.fromBytes(Vectors.toBytes(single)));
        }

        @Test
        void toBytesFromBytesNaN() {
            float[] withNan = {1.0f, Float.NaN, 3.0f};
            float[] restored = Vectors.fromBytes(Vectors.toBytes(withNan));
            assertEquals(1.0f, restored[0]);
            assertTrue(Float.isNaN(restored[1]), "NaN should survive round-trip");
            assertEquals(3.0f, restored[2]);
        }

        @Test
        void fromBytesNullReturnsEmpty() {
            float[] result = Vectors.fromBytes(null);
            assertEquals(0, result.length);
        }

        @Test
        void toBytesFromBytesRealistic768Dim() {
            Random rng = new Random(99);
            float[] original = randomVector(768, rng);
            float[] restored = Vectors.fromBytes(Vectors.toBytes(original));
            assertArrayEquals(original, restored);
        }

        private float[] randomVector(int dim, Random rng) {
            float[] v = new float[dim];
            for (int i = 0; i < dim; i++) v[i] = rng.nextFloat() * 2 - 1;
            return v;
        }
    }

    // ---------------------------------------------------------------
    //  Fts
    // ---------------------------------------------------------------
    @Nested
    class FtsTest {

        @Test
        void sanitiseMatchNormalWords() {
            String result = Fts.sanitiseMatch("hello world");
            // Both words are >= 2 chars, should appear quoted and OR-joined
            assertTrue(result.contains("\"hello\""), "got: " + result);
            assertTrue(result.contains("\"world\""), "got: " + result);
            assertTrue(result.contains(" OR "), "got: " + result);
        }

        @Test
        void sanitiseMatchStripsSpecialChars() {
            String result = Fts.sanitiseMatch("hello! @world# $test%");
            // Special chars become spaces; tokens remain
            assertTrue(result.contains("\"hello\""), "got: " + result);
            assertTrue(result.contains("\"world\""), "got: " + result);
            assertTrue(result.contains("\"test\""), "got: " + result);
            // None of the special characters should leak through
            assertFalse(result.contains("!"));
            assertFalse(result.contains("@"));
            assertFalse(result.contains("#"));
            assertFalse(result.contains("$"));
            assertFalse(result.contains("%"));
        }

        @Test
        void sanitiseMatchNullReturnsSafeExpression() {
            String result = Fts.sanitiseMatch(null);
            assertNotNull(result);
            assertFalse(result.isEmpty(), "null should not produce empty string");
            // Should be the empty-quoted fallback
            assertEquals("\"\"", result);
        }

        @Test
        void sanitiseMatchEmptyReturnsSafeExpression() {
            String result = Fts.sanitiseMatch("");
            assertNotNull(result);
            assertEquals("\"\"", result);
        }

        @Test
        void sanitiseMatchSingleCharTokensDropped() {
            // "I a b" — all tokens are single-char and should be dropped
            String result = Fts.sanitiseMatch("I a b");
            assertEquals("\"\"", result, "single-char tokens should be dropped");
        }

        @Test
        void sanitiseMatchMixOfShortAndLongTokens() {
            String result = Fts.sanitiseMatch("I am here");
            // "I" dropped (1 char), "am" kept, "here" kept
            assertTrue(result.contains("\"am\""), "got: " + result);
            assertTrue(result.contains("\"here\""), "got: " + result);
            assertFalse(result.contains("\"I\""), "'I' should be dropped");
        }

        @Test
        void sanitiseMatchUnicode() {
            // Unicode letters should be preserved (isLetterOrDigit)
            String result = Fts.sanitiseMatch("caf\u00e9 na\u00efve");
            assertTrue(result.contains("caf\u00e9"), "got: " + result);
            assertTrue(result.contains("na\u00efve"), "got: " + result);
        }

        @Test
        void sanitiseMatchPreservesHyphenAndUnderscore() {
            String result = Fts.sanitiseMatch("well-known data_set");
            // Hyphens and underscores are kept by the implementation
            assertTrue(result.contains("well-known"), "got: " + result);
            assertTrue(result.contains("data_set"), "got: " + result);
        }
    }

    // ---------------------------------------------------------------
    //  Tags
    // ---------------------------------------------------------------
    @Nested
    class TagsTest {

        @Test
        void encodeDecodeRoundTrip() {
            Map<String, String> original = new LinkedHashMap<>();
            original.put("type", "fact");
            original.put("source", "chat");
            String encoded = Tags.encode(original);
            Map<String, String> decoded = Tags.decode(encoded);
            assertEquals(original, decoded);
        }

        @Test
        void encodeEmptyMapReturnsNull() {
            assertNull(Tags.encode(Map.of()));
        }

        @Test
        void encodeNullMapReturnsNull() {
            assertNull(Tags.encode(null));
        }

        @Test
        void decodeNullReturnsEmptyMap() {
            Map<String, String> result = Tags.decode(null);
            assertTrue(result.isEmpty());
        }

        @Test
        void decodeEmptyStringReturnsEmptyMap() {
            Map<String, String> result = Tags.decode("");
            assertTrue(result.isEmpty());
        }

        @Test
        void encodeDecodeMultipleEntries() {
            Map<String, String> tags = new LinkedHashMap<>();
            tags.put("a", "1");
            tags.put("b", "2");
            tags.put("c", "3");
            String encoded = Tags.encode(tags);
            Map<String, String> decoded = Tags.decode(encoded);
            assertEquals(3, decoded.size());
            assertEquals("1", decoded.get("a"));
            assertEquals("2", decoded.get("b"));
            assertEquals("3", decoded.get("c"));
        }

        @Test
        void encodeStripsDelimitersFromKeysAndValues() {
            Map<String, String> tags = new LinkedHashMap<>();
            tags.put("key;with;semi", "val=with=eq");
            String encoded = Tags.encode(tags);
            // Semicolons and equals in keys/values are stripped
            assertFalse(encoded.contains("key;with"), "semicolons in key should be stripped");
            Map<String, String> decoded = Tags.decode(encoded);
            assertEquals(1, decoded.size());
            assertEquals("valwitheq", decoded.get("keywithsemi"));
        }

        @Test
        void encodeHandlesNullValue() {
            Map<String, String> tags = new LinkedHashMap<>();
            tags.put("key", null);
            String encoded = Tags.encode(tags);
            assertNotNull(encoded);
            Map<String, String> decoded = Tags.decode(encoded);
            assertEquals("", decoded.get("key"));
        }

        @Test
        void decodeIgnoresMalformedPieces() {
            // A piece without '=' should be skipped
            Map<String, String> decoded = Tags.decode("good=val;bad;also=ok");
            assertEquals(2, decoded.size());
            assertEquals("val", decoded.get("good"));
            assertEquals("ok", decoded.get("also"));
            assertFalse(decoded.containsKey("bad"));
        }
    }

    // ---------------------------------------------------------------
    //  Tokens
    // ---------------------------------------------------------------
    @Nested
    class TokensTest {

        @Test
        void countNullReturnsZero() {
            assertEquals(0, Tokens.count(null));
        }

        @Test
        void countEmptyReturnsZero() {
            assertEquals(0, Tokens.count(""));
        }

        @Test
        void countSingleWord() {
            int count = Tokens.count("hello");
            // "hello" = 5 chars -> chars/4 = 1(.25 rounded), word count = 1 -> max(1,1) = 1
            assertTrue(count >= 1 && count <= 2, "single word should be ~1, got " + count);
        }

        @Test
        void countHelloWorld() {
            int count = Tokens.count("hello world");
            // 11 chars -> chars/4 ~ 3, words = 2 -> max(2,3) = 3
            assertTrue(count >= 2 && count <= 4,
                    "\"hello world\" should be ~2-3 tokens, got " + count);
        }

        @Test
        void countLongTextRoughlyCharsDiv4() {
            String text = "The quick brown fox jumps over the lazy dog. "
                    + "This sentence is used to test heuristic token counting "
                    + "and should produce a count roughly equal to the character "
                    + "length divided by four.";
            int count = Tokens.count(text);
            int expected = Math.round(text.length() / 4.0f);
            // Should be within ~25% of chars/4
            assertTrue(count >= expected * 0.75 && count <= expected * 1.25,
                    "long text: expected ~" + expected + ", got " + count);
        }

        @Test
        void countWordsFloorKicksInForShortTokens() {
            // Many short words: "a b c d e" -> 9 chars / 4 = 2, words = 5 -> max(5,2) = 5
            int count = Tokens.count("a b c d e");
            assertEquals(5, count, "word count floor should kick in");
        }

        @Test
        void countHandlesNewlinesAndTabs() {
            // Newlines and tabs count as word separators
            int count = Tokens.count("hello\nworld\tfoo");
            // 3 words, 15 chars -> chars/4 = 4 -> max(3,4) = 4
            assertTrue(count >= 3, "should count at least 3 words, got " + count);
        }
    }
}
