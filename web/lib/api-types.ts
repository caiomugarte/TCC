export type GenericProfile = "conservador" | "moderado" | "arrojado";
export type Plan = "basic" | "premium";
export type EntitlementStatus = "active" | "inactive" | "grace_period";

export type AssetClassKey =
  | "brazilian_stocks"
  | "fiis"
  | "international"
  | "fixed_income"
  | "crypto";

export type Account = {
  id: string;
  email: string | null;
  plan: Plan;
  entitlementStatus: EntitlementStatus;
};

export type ProfileInput = {
  answers: Record<string, string | string[]>;
  investableCapitalBrl: number;
  consented: boolean;
};

export type Profile = ProfileInput & {
  id: string;
  accountId: string;
  version: number;
  dimensions: Record<string, number>;
  suitabilityScore: number;
  genericProfile: GenericProfile;
  consentedAt: string;
  createdAt: string;
};

export type RecommendationRequest = {
  profileId?: string;
  investableCapitalBrl?: number;
};

export type PremiumRecommendationRequest = {
  profileId?: string;
  force?: boolean;
};

export type RecommendationStatus = "queued" | "running" | "completed" | "failed";

export type AllocationClass = {
  key: AssetClassKey;
  label: string;
  targetWeight: number;
  targetAmountBrl: number;
  metrics?: Record<string, unknown>;
};

export type RecommendationConstituent = {
  ticker: string;
  sleeveWeight: number;
  portfolioWeight: number;
  targetAmountBrl: number;
  reasons: string[];
};

export type RecommendationProvenance = Record<string, unknown> & {
  selector_sources?: Partial<Record<"stocks" | "fiis", {
    provider?: string;
    retrieved_at?: string;
    sha256?: string;
  }>>;
};

export type Recommendation = {
  id: string;
  profileVersion: number;
  plan: Plan;
  modelVersion: string;
  snapshotId: string;
  snapshotCutoff: string;
  classes: AllocationClass[];
  assumptions: string[];
  risks: string[];
  createdAt: string;
  status: RecommendationStatus;
  startedAt: string | null;
  completedAt: string | null;
  failureCode: string | null;
  failureMessage: string | null;
  stocks: RecommendationConstituent[];
  fiis: RecommendationConstituent[];
  policy: Record<string, unknown> | null;
  provenance: RecommendationProvenance | null;
};

export type PremiumRecommendation = Recommendation;

export type PortfolioInput = {
  currency: "BRL";
  classes: Partial<Record<AssetClassKey, number>>;
};

export type PortfolioSnapshot = PortfolioInput & {
  id: string;
  source: "manual";
  capturedAt: string;
  totalValueBrl: number;
  normalizedWeights: Partial<Record<AssetClassKey, number>>;
};

export type DriftStatus = "within_range" | "underweight" | "overweight";
export type SuggestedAction = "hold" | "contribute" | "review_sale";

export type DriftItem = {
  classKey: AssetClassKey;
  currentWeight: number;
  targetWeight: number;
  drift: number;
  valueGapBrl: number;
  status: DriftStatus;
  suggestedAction: SuggestedAction;
};

export type Review = {
  recommendationId: string;
  portfolioId: string;
  driftBand: number;
  items: DriftItem[];
};

export type ApiErrorPayload = {
  code?: string;
  message?: string;
  details?: unknown;
};
