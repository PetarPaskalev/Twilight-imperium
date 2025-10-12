export const metadata = {
  title: "Twilight Imperium Assistant",
  description: "Ask about TI4 rules and factions",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: 'system-ui, sans-serif' }}>{children}</body>
    </html>
  );
}


