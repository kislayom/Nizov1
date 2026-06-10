package ai.nizo.memory.vector;

import java.util.List;

/**
 * Pluggable vector similarity index. Implementations must partition by userId
 * so that topK queries never cross user boundaries.
 */
public interface VectorIndex {
    void add(String userId, String id, float[] vector);
    void remove(String userId, String id);
    int size();
    List<Hit> topK(String userId, float[] query, int k);

    record Hit(String id, double score) {}
}
