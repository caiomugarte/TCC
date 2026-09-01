import { redirect } from "next/navigation";

import { OnboardingQuestionnaire } from "@/components/onboarding/questionnaire";
import { getProfile } from "@/lib/api-client";
import { requireSession } from "@/lib/route-guards";

export default async function OnboardingPage() {
  const profile = await getProfile(await requireSession());
  if (profile) redirect("/app/recommendation");

  return <OnboardingQuestionnaire />;
}
