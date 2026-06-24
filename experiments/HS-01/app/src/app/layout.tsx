import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HS-01 Human Validity Study",
  description: "VLM boundary item human validity study",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
