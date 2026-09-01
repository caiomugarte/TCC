import { RecommendationView } from "@/components/recommendation/recommendation-view";
import { requireProfile } from "@/lib/route-guards";

export default async function RecommendationPage() {
  await requireProfile();
  return <RecommendationView />;
}
