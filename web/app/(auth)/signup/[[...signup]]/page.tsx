import { SignUp } from "@clerk/nextjs";

export default function SignupPage() {
  return (
    <div className="clerk-card">
      <SignUp fallbackRedirectUrl="/app/recommendation" signInUrl="/login" />
    </div>
  );
}
