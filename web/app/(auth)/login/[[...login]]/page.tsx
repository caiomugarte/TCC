import { SignIn } from "@clerk/nextjs";

export default function LoginPage() {
  return (
    <div className="clerk-card">
      <SignIn fallbackRedirectUrl="/app/recommendation" signUpUrl="/signup" />
    </div>
  );
}
