import { UserButton } from "@clerk/nextjs";
import Link from "next/link";

export function ProtectedNav() {
  return (
    <header className="protected-nav">
      <nav className="nav" aria-label="Navegação da aplicação">
        <Link className="brand" href="/app/onboarding" aria-label="Prumo, aplicação">
          <svg className="logo-mark" viewBox="0 0 64 64" aria-hidden="true">
            <path
              d="M22 8h14c11 0 20 9 20 20s-9 20-20 20H30v8a8 8 0 1 1-16 0V16a8 8 0 0 1 8-8Z"
              fill="currentColor"
            />
            <path d="M30 22h6a6 6 0 0 1 0 12h-6V22Z" fill="#f7f7f4" />
          </svg>
          <span>prumo</span>
        </Link>
        <div className="nav-links">
          <Link href="/app/recommendation">Recomendação</Link>
          <Link href="/app/portfolio">Carteira</Link>
          <Link href="/app/review">Revisão</Link>
          <Link href="/account">Conta</Link>
          <UserButton />
        </div>
      </nav>
    </header>
  );
}
