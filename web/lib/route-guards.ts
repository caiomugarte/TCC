import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";

import {
  ApiClientError,
  getLatestRecommendation,
  getPortfolio,
  getProfile,
} from "@/lib/api-client";

export async function requireSession(): Promise<string> {
  const { isAuthenticated, getToken } = await auth();
  if (!isAuthenticated) redirect("/login");

  const token = await getToken();
  if (!token) redirect("/login");
  return token;
}

async function readGuarded<T>(token: string, read: (token: string) => Promise<T>): Promise<T> {
  try {
    return await read(token);
  } catch (error) {
    if (error instanceof ApiClientError && error.status === 401) redirect("/login");
    throw error;
  }
}

export async function requireProfile(): Promise<string> {
  const token = await requireSession();
  const profile = await readGuarded(token, getProfile);
  if (!profile) redirect("/app/onboarding");
  return token;
}

export async function requireRecommendation(token: string): Promise<void> {
  const recommendation = await readGuarded(token, getLatestRecommendation);
  if (!recommendation) redirect("/app/recommendation");
}

export async function requirePortfolio(token: string): Promise<void> {
  const portfolio = await readGuarded(token, getPortfolio);
  if (!portfolio) redirect("/app/portfolio");
}
