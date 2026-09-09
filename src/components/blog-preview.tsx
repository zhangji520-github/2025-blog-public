'use client'

import { motion } from 'motion/react'
import { INIT_DELAY } from '@/consts'
import { useMarkdownRender } from '@/hooks/use-markdown-render'
import { useSize } from '@/hooks/use-size'
import { BlogSidebar } from '@/components/blog-sidebar'
import { useConfigStore } from '@/app/(home)/stores/config-store'

type BlogPreviewProps = {
	markdown: string
	title: string
	tags: string[]
	date: string
	summary?: string
	cover?: string
	slug?: string
}

export function BlogPreview({ markdown, title, tags, date, summary, cover, slug }: BlogPreviewProps) {
	const { maxSM: isMobile } = useSize()
	// The article header already displays the title. Keep the source Markdown intact.
	const openingHeading = markdown.match(/^\uFEFF?\s*#\s+([^\r\n]+)\r?\n/)
	const body = openingHeading?.[1].trim() === title.trim() ? markdown.slice(openingHeading[0].length).replace(/^\s*\n/, '') : markdown
	const { content, toc, loading } = useMarkdownRender(body)
	const { siteContent } = useConfigStore()
	const summaryInContent = siteContent.summaryInContent ?? false

	if (loading) {
		return <div className='text-secondary flex h-full items-center justify-center text-sm'>渲染中...</div>
	}

	return (
		<div className='mx-auto flex max-w-[1140px] justify-center gap-6 px-6 pt-28 pb-12 max-sm:px-0'>
			<motion.article
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ delay: INIT_DELAY }}
				className='card bg-article static min-w-0 flex-1 overflow-auto rounded-2xl p-10 max-sm:p-5'>
				<div>
					<h1 className='text-center text-3xl leading-snug font-bold tracking-tight max-sm:text-2xl'>{title}</h1>

					<div className='text-secondary mt-4 flex flex-wrap items-center justify-center gap-3 px-8 text-center text-sm'>
						{tags.map(t => (
							<span key={t}>#{t}</span>
						))}
					</div>

					<div className='text-secondary mt-3 text-center text-sm'>{date}</div>

					{summary && summaryInContent && <div className='text-secondary mt-6 cursor-text text-center text-sm'>“{summary}”</div>}

					<div className='prose mt-8 max-w-none cursor-text border-t border-black/5 pt-6'>{content}</div>
				</div>
			</motion.article>

			{!isMobile && <BlogSidebar cover={cover} summary={summary} toc={toc} slug={slug} />}
		</div>
	)
}
