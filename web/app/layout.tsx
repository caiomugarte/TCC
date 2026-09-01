import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";

import "./globals.css";

export const metadata: Metadata = {
  title: "Prumo",
  description: "Decisões mais claras para sua carteira.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body><ClerkProvider>{children}</ClerkProvider></body>
    </html>
  );
}
