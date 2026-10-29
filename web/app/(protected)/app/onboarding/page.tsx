import { redirect } from "next/navigation";

import { OnboardingQuestionnaire } from "@/components/onboarding/questionnaire";
import { getProfile } from "@/lib/api-client";
import { requireSession } from "@/lib/route-guards";

export default async function OnboardingPage({
  searchParams,
}: {
  searchParams: Promise<{ edit?: string }>;
}) {
  const { edit } = await searchParams;
  const profile = await getProfile(await requireSession());
  if (profile && edit !== "1") redirect("/app/recommendation");

  return <OnboardingQuestionnaire />;
}
