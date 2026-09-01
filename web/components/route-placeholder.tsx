import Link from "next/link";

type RoutePlaceholderProps = {
  eyebrow: string;
  title: string;
  description: string;
  nextHref?: string;
  nextLabel?: string;
};

export function RoutePlaceholder({
  eyebrow,
  title,
  description,
  nextHref,
  nextLabel,
}: RoutePlaceholderProps) {
  return (
    <section className="card">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p>{description}</p>
      {nextHref && nextLabel ? (
        <div className="actions">
          <Link className="button" href={nextHref}>{nextLabel}</Link>
        </div>
      ) : null}
    </section>
  );
}
