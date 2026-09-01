import { ReviewView } from "@/components/review/review-view";
import { requirePortfolio, requireProfile, requireRecommendation } from "@/lib/route-guards";

export default async function ReviewPage() {
  const token = await requireProfile();
  await requireRecommendation(token);
  await requirePortfolio(token);
  return <ReviewView />;
}
