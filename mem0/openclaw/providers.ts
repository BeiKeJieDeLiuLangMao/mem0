/**
 * Mem0 provider implementations: Pure TypeScript OSS implementation.
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type {
  Mem0Config,
  Mem0Provider,
} from "./types.ts";
import { OSSProvider } from "./oss-provider.ts";

// ============================================================================
// Provider Factory
// ============================================================================

export function createProvider(
  cfg: Mem0Config,
  _api: OpenClawPluginApi,
): Mem0Provider {
  // Always use pure TypeScript OSS implementation
  return new OSSProvider(cfg.oss);
}
