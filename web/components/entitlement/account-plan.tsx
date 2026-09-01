import type { Account } from "@/lib/api-types";

function statusLabel(status: Account["entitlementStatus"]): string {
  return status === "active" ? "Ativo" : status === "grace_period" ? "Período de tolerância" : "Inativo";
}

export function AccountPlan({ account }: { account: Account }) {
  const premium = account.plan === "premium";

  return (
    <section className="stack" aria-labelledby="account-title">
      <div>
        <p className="eyebrow">Conta</p>
        <h1 id="account-title">Plano e acesso</h1>
        <p>{account.email ?? "Conta autenticada via Clerk."}</p>
      </div>

      <div className="card-grid">
        <article className="card">
          <p className="eyebrow">Plano atual</p>
          <h2>{premium ? "Premium" : "Basic"}</h2>
          <p>Status do entitlement: {statusLabel(account.entitlementStatus)}.</p>
        </article>

        <article className="card locked-card" aria-labelledby="premium-access-title">
          <p className="eyebrow">Recursos Premium</p>
          <h2 id="premium-access-title">{premium ? "Acesso liberado" : "Acesso bloqueado"}</h2>
          <p>
            {premium
              ? "O servidor confirmou seu entitlement Premium."
              : "O acesso só será liberado após confirmação do servidor."}
          </p>
        </article>
      </div>
    </section>
  );
}
