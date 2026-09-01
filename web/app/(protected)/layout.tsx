import { ProtectedNav } from "@/components/protected-nav";

export default function ProtectedLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="protected-shell">
      <ProtectedNav />
      <main className="page protected-content">{children}</main>
    </div>
  );
}
