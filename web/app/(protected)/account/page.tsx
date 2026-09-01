import { AccountPlan } from "@/components/entitlement/account-plan";
import { getMe } from "@/lib/api-client";
import { requireSession } from "@/lib/route-guards";

export default async function AccountPage() {
  const account = await getMe(await requireSession());
  return <AccountPlan account={account} />;
}
