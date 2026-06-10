package ai.nizo.memory.config;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reads {@link NizoConfig} from a YAML file. Translation rules:
 *
 * <ul>
 *   <li>Snake-case keys in the file map to camelCase Java fields via
 *       {@link PropertyNamingStrategies#SNAKE_CASE}.</li>
 *   <li>A missing file returns {@link NizoConfig#defaults()} — services start
 *       with sane defaults rather than blowing up.</li>
 *   <li>Unknown keys are ignored (forward compat).</li>
 *   <li>Sections omitted from the YAML are filled in from
 *       {@link NizoConfig#withDefaults()} after parse.</li>
 * </ul>
 */
public final class ConfigLoader {

    private static final Logger log = LoggerFactory.getLogger(ConfigLoader.class);

    private static final ObjectMapper YAML = new ObjectMapper(new YAMLFactory())
            .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    private ConfigLoader() {}

    /**
     * Load the config from {@code yamlPath}.
     *
     * @throws ConfigException if the file exists but cannot be parsed.
     */
    public static NizoConfig load(Path yamlPath) {
        if (yamlPath == null || !Files.isRegularFile(yamlPath)) {
            log.info("No config file at {} — using defaults", yamlPath);
            return NizoConfig.defaults();
        }
        try {
            NizoConfig parsed = YAML.readValue(yamlPath.toFile(), NizoConfig.class);
            if (parsed == null) {
                log.info("Config file {} is empty — using defaults", yamlPath);
                return NizoConfig.defaults();
            }
            log.info("Loaded config from {}", yamlPath);
            return parsed.withDefaults();
        } catch (IOException e) {
            throw new ConfigException(
                    "Failed to parse config file " + yamlPath + ": " + e.getMessage(), e);
        }
    }

    /**
     * Same as {@link #load(Path)} but never throws — parse errors are logged
     * and {@link NizoConfig#defaults()} is returned. Use this in main() to
     * keep the service starting on a malformed config.
     */
    public static NizoConfig loadOrDefault(Path yamlPath) {
        try {
            return load(yamlPath);
        } catch (RuntimeException e) {
            log.warn("Config load failed, falling back to defaults: {}", e.getMessage());
            return NizoConfig.defaults();
        }
    }

    /** Thrown for unrecoverable parse failures. */
    public static final class ConfigException extends RuntimeException {
        public ConfigException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
