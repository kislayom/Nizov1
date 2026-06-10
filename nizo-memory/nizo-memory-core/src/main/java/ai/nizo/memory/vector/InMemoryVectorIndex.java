package ai.nizo.memory.vector;

import ai.nizo.memory.api.memory.MemoryItem;
import ai.nizo.memory.util.Vectors;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Simple but fast in-memory brute-force cosine index, partitioned by userId.
 *
 * <p>For 192 GB RAM hosts this easily scales to millions of 768-dim vectors
 * (~3 GB per 1 M). When the corpus outgrows RAM we can swap this with a
 * pluggable HNSW implementation behind the {@link VectorIndex} interface.
 *
 * <p>Each userId gets its own vector map, so topK queries never cross user
 * boundaries. The {@link #add} and {@link #topK} methods require a userId
 * parameter to enforce this isolation.
 */
public final class InMemoryVectorIndex implements VectorIndex {

    /** userId → (memoryId → vector) */
    private final Map<String, Map<String, float[]>> partitions = new ConcurrentHashMap<>();

    @Override
    public void add(String userId, String id, float[] vector) {
        if (vector == null) return;
        String uid = userId == null ? "default" : userId;
        partitions.computeIfAbsent(uid, k -> new ConcurrentHashMap<>()).put(id, vector);
    }

    @Override
    public void remove(String userId, String id) {
        String uid = userId == null ? "default" : userId;
        Map<String, float[]> partition = partitions.get(uid);
        if (partition != null) {
            partition.remove(id);
            if (partition.isEmpty()) partitions.remove(uid);
        }
    }

    @Override
    public int size() {
        int total = 0;
        for (Map<String, float[]> p : partitions.values()) total += p.size();
        return total;
    }

    /** Number of vectors for a specific user. */
    public int size(String userId) {
        String uid = userId == null ? "default" : userId;
        Map<String, float[]> partition = partitions.get(uid);
        return partition == null ? 0 : partition.size();
    }

    /** Drop the entire partition for a user — GDPR forget-user cascade. */
    public void removeAllForUser(String userId) {
        String uid = userId == null ? "default" : userId;
        partitions.remove(uid);
    }

    @Override
    public List<Hit> topK(String userId, float[] query, int k) {
        String uid = userId == null ? "default" : userId;
        Map<String, float[]> partition = partitions.get(uid);
        if (query == null || partition == null || partition.isEmpty()) return List.of();

        PriorityQueue<Hit> heap = new PriorityQueue<>(Comparator.comparingDouble(Hit::score));
        for (Map.Entry<String, float[]> e : partition.entrySet()) {
            double s = Vectors.cosine(query, e.getValue());
            if (heap.size() < k) {
                heap.offer(new Hit(e.getKey(), s));
            } else if (heap.peek().score() < s) {
                heap.poll();
                heap.offer(new Hit(e.getKey(), s));
            }
        }
        List<Hit> out = new ArrayList<>(heap);
        out.sort(Comparator.comparingDouble(Hit::score).reversed());
        return out;
    }

    /** Warm from an existing store on startup. */
    public void hydrate(Iterable<MemoryItem> items) {
        for (MemoryItem m : items) {
            if (m.embedding() != null) {
                add(m.userId(), m.id(), m.embedding());
            }
        }
    }
}
