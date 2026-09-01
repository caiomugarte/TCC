import Link from "next/link";

export default function AuthLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <main className="page">
      <header className="nav">
        <Link className="brand" href="/">Prumo</Link>
        <Link href="/">Voltar para início</Link>
      </header>
      <div className="form-card">{children}</div>
    </main>
  );
}
