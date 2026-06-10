package ai.nizo.tools.net;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetAddress;
import java.net.URI;
import java.net.UnknownHostException;
import java.util.Locale;
import java.util.Set;

/**
 * Server-Side Request Forgery guard.
 *
 * <p>Every URL the agent fetches outbound from {@code WebFetchTool}, {@code HttpJsonTool},
 * {@code SmartProxyClient}, etc. should pass through {@link #assertSafe(URI)} first. The agent
 * is steerable by an LLM with web-search context — it cannot be trusted to NOT be tricked into
 * fetching e.g. {@code http://169.254.169.254/latest/meta-data/iam/security-credentials/} (AWS
 * IMDS) or {@code http://10.0.0.1/internal-admin}. This class is the choke point that says no.
 *
 * <h2>What gets blocked</h2>
 * <ul>
 *   <li>Loopback: {@code 127.0.0.0/8}, {@code ::1}, the literal hostname {@code localhost}.</li>
 *   <li>Link-local: {@code 169.254.0.0/16} (covers AWS/GCP/Azure metadata at
 *       {@code 169.254.169.254}) and IPv6 {@code fe80::/10}.</li>
 *   <li>Site-local / RFC1918 private: {@code 10.0.0.0/8}, {@code 172.16.0.0/12},
 *       {@code 192.168.0.0/16}, and IPv6 site-local {@code fec0::/10}.</li>
 *   <li>Wildcard / "any" addresses: {@code 0.0.0.0}, {@code ::}.</li>
 *   <li>Multicast: {@code 224.0.0.0/4}, {@code ff00::/8}.</li>
 *   <li>Cloud-metadata literals by hostname: {@code metadata.google.internal},
 *       {@code instance-data} (used by some EC2 setups).</li>
 * </ul>
 *
 * <h2>Escape hatch</h2>
 * The agent runs alongside {@code llama-server} on {@code localhost:8080} in dev. Setting
 * {@code NIZO_NET_ALLOW_LOOPBACK=1} disables the loopback check (only loopback — link-local,
 * site-local, multicast, and metadata hostnames remain blocked). Off by default.
 *
 * <h2>What this is NOT</h2>
 * Not a URL allowlist. Not a content-filter. The check is purely <em>where the network
 * request will land</em>: hostnames are resolved and every returned {@link InetAddress}
 * must be public. We don't second-guess the path or the response body.
 *
 * <p>Method is package-public-final and stateless — safe to call from any thread.
 */
public final class SsrfGuard {

    private static final Logger LOG = LoggerFactory.getLogger(SsrfGuard.class);

    /** Hostnames that resolve to public IPs but are still off-limits (cloud metadata). */
    private static final Set<String> BLOCKED_HOSTS = Set.of(
            "localhost",
            "metadata.google.internal",
            "instance-data",
            "instance-data.ec2.internal");

    /** {@code NIZO_NET_ALLOW_LOOPBACK=1} → permit loopback (dev convenience for llama-server). */
    private static final boolean ALLOW_LOOPBACK =
            "1".equals(System.getenv("NIZO_NET_ALLOW_LOOPBACK"))
            || "true".equalsIgnoreCase(System.getenv("NIZO_NET_ALLOW_LOOPBACK"));

    private SsrfGuard() { /* no instances */ }

    /**
     * Validate the URL before issuing a request.
     *
     * @param url the target URL — must already be parsed (so callers see {@link URI}-level errors first)
     * @throws SecurityException with a message naming the offending host, suitable for surfacing
     *     verbatim to the LLM as a tool error
     */
    public static void assertSafe(URI url) {
        if (url == null) throw new SecurityException("blocked: null URL");
        String host = url.getHost();
        if (host == null || host.isBlank()) {
            throw new SecurityException("blocked: URL has no host: " + url);
        }
        String lower = host.toLowerCase(Locale.ROOT);

        // Always block hostname literals regardless of resolution outcome.
        if (BLOCKED_HOSTS.contains(lower) && !ALLOW_LOOPBACK) {
            throw new SecurityException("blocked: hostname '" + host
                    + "' resolves to internal/metadata target");
        }
        // Even with loopback allowed, still block non-loopback metadata hosts.
        if (lower.equals("metadata.google.internal")
                || lower.equals("instance-data")
                || lower.equals("instance-data.ec2.internal")) {
            throw new SecurityException("blocked: hostname '" + host
                    + "' is a cloud-metadata endpoint");
        }

        InetAddress[] addrs;
        try {
            addrs = InetAddress.getAllByName(host);
        } catch (UnknownHostException uhe) {
            // Don't leak DNS detail to the LLM — but DO refuse the call.
            throw new SecurityException("blocked: cannot resolve host '" + host + "'");
        }
        if (addrs == null || addrs.length == 0) {
            throw new SecurityException("blocked: no addresses for host '" + host + "'");
        }
        for (InetAddress addr : addrs) {
            String reason = classify(addr);
            if (reason == null) continue; // safe
            // Loopback is the only category the dev escape hatch unlocks.
            if ("loopback".equals(reason) && ALLOW_LOOPBACK) {
                LOG.debug("loopback permitted by NIZO_NET_ALLOW_LOOPBACK: {}", addr);
                continue;
            }
            throw new SecurityException("blocked: host '" + host + "' resolves to "
                    + reason + " address " + addr.getHostAddress());
        }
    }

    /**
     * Return the category name if the address is unsafe, or {@code null} if it's a public address.
     * Package-private for unit testing.
     */
    static String classify(InetAddress addr) {
        if (addr == null) return "null";
        if (addr.isAnyLocalAddress())     return "wildcard";       // 0.0.0.0 / ::
        if (addr.isLoopbackAddress())     return "loopback";       // 127.0.0.0/8 / ::1
        if (addr.isLinkLocalAddress())    return "link-local";     // 169.254.0.0/16 / fe80::
        if (addr.isSiteLocalAddress())    return "site-local";     // RFC1918 IPv4 / fec0::
        if (addr.isMulticastAddress())    return "multicast";      // 224/4 / ff00::
        // IPv4 RFC1918 covers most internal — but the JDK's site-local check handles that.
        // IPv4-mapped IPv6 (::ffff:a.b.c.d) — InetAddress.getAddress() returns 4 bytes already.
        return null;
    }
}
