import { PortfolioForm } from "@/components/portfolio/portfolio-form";
import { requireProfile, requireRecommendation } from "@/lib/route-guards";

export default async function PortfolioPage() {
  const token = await requireProfile();
  await requireRecommendation(token);
  return <PortfolioForm />;
}
