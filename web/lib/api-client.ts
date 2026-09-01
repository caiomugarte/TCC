import type {
  Account,
  ApiErrorPayload,
  PortfolioInput,
  PortfolioSnapshot,
  Profile,
  ProfileInput,
  Recommendation,
  RecommendationRequest,
  Review,
} from "@/lib/api-types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
type AuthToken = string | null | undefined;

export class ApiClientError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;

  constructor(status: number, payload: ApiErrorPayload, fallbackMessage: string) {
    super(payload.message ?? fallbackMessage);
    this.name = "ApiClientError";
    this.status = status;
    this.code = payload.code ?? "api_error";
    this.details = payload.details;
  }
}

async function readPayload(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return undefined;

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { message: text } satisfies ApiErrorPayload;
  }
}

function asErrorPayload(payload: unknown): ApiErrorPayload {
  if (!payload || typeof payload !== "object") return {};

  const candidate = payload as Record<string, unknown>;
  const detail = candidate.detail;
  const source = detail && typeof detail === "object"
    ? detail as Record<string, unknown>
    : candidate;
  return {
    code: typeof source.code === "string" ? source.code : undefined,
    message: typeof source.message === "string" ? source.message : undefined,
    details: source.details ?? (Array.isArray(detail) ? detail : undefined),
  };
}

async function request<T>(path: string, init: RequestInit = {}, token?: AuthToken): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
    credentials: "include",
  });
  const payload = await readPayload(response);

  if (!response.ok) {
    throw new ApiClientError(
      response.status,
      asErrorPayload(payload),
      response.statusText || "A solicitação não pôde ser concluída.",
    );
  }

  return payload as T;
}

export function getMe(token?: AuthToken): Promise<Account> {
  return request<Account>("/v1/me", {}, token);
}

export function getProfile(token?: AuthToken): Promise<Profile | null> {
  return request<Profile | null>("/v1/profile", {}, token);
}

export function saveProfile(input: ProfileInput, token?: AuthToken): Promise<Profile> {
  return request<Profile>("/v1/profile", {
    method: "PUT",
    body: JSON.stringify(input),
  }, token);
}

export function createRecommendation(
  input: RecommendationRequest = {},
  token?: AuthToken,
): Promise<Recommendation> {
  return request<Recommendation>("/v1/recommendations", {
    method: "POST",
    body: JSON.stringify(input),
  }, token);
}

export function getLatestRecommendation(token?: AuthToken): Promise<Recommendation | null> {
  return request<Recommendation | null>("/v1/recommendations", {}, token);
}

export function savePortfolio(input: PortfolioInput, token?: AuthToken): Promise<PortfolioSnapshot> {
  return request<PortfolioSnapshot>("/v1/portfolio", {
    method: "PUT",
    body: JSON.stringify(input),
  }, token);
}

export function getPortfolio(token?: AuthToken): Promise<PortfolioSnapshot | null> {
  return request<PortfolioSnapshot | null>("/v1/portfolio", {}, token);
}

export function getReview(token?: AuthToken): Promise<Review> {
  return request<Review>("/v1/review", {}, token);
}
