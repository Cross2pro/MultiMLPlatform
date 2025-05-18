import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "多模型机器学习预测系统",
  description: "基于Next.js和Express构建的多模型机器学习预测系统",
  authors: [{ name: "李嘉俊", url: "https://github.com" }],
  creator: "长沙理工大学 李嘉俊",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
