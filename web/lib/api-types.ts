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

export type AllocationClass = {
  key: AssetClassKey;
  label: string;
  targetWeight: number;
  targetAmountBrl: number;
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
};

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
