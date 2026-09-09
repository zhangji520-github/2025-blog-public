import { useEffect, useState, type ReactElement } from 'react'
import parse, { domToReact, type HTMLReactParserOptions, Element, type DOMNode } from 'html-react-parser'
import { renderMarkdown, type TocItem } from '@/lib/markdown-renderer'
import { MarkdownImage } from '@/components/markdown-image'
import { CodeBlock } from '@/components/code-block'

type MarkdownRenderResult = {
	content: ReactElement | null
	toc: TocItem[]
	loading: boolean
}

export function useMarkdownRender(markdown: string): MarkdownRenderResult {
	const [content, setContent] = useState<ReactElement | null>(null)
	const [toc, setToc] = useState<TocItem[]>([])
	const [loading, setLoading] = useState<boolean>(true)

	useEffect(() => {
		let cancelled = false

		async function render() {
			setLoading(true)
			try {
				const { html, toc } = await renderMarkdown(markdown)
				if (!cancelled) {
					const options: HTMLReactParserOptions = {
						replace(domNode: DOMNode) {
							if (!(domNode instanceof Element)) return
							if (domNode.name === 'pre' && domNode.attribs['data-code'] !== undefined) {
								return <CodeBlock code={domNode.attribs['data-code']}><pre className={domNode.attribs.class}>{domToReact(domNode.children as DOMNode[])}</pre></CodeBlock>
							}
							if (domNode.name === 'img') {
								const { src, alt, title } = domNode.attribs
								return <MarkdownImage src={src} alt={alt} title={title} />
							}
						}
					}
					const reactContent = parse(html, options) as ReactElement
					setContent(reactContent)
					setToc(toc)
				}
			} catch (error) {
				console.error('Markdown render error:', error)
				if (!cancelled) {
					setContent(null)
					setToc([])
				}
			} finally {
				if (!cancelled) {
					setLoading(false)
				}
			}
		}

		render()

		return () => {
			cancelled = true
		}
	}, [markdown])

	return { content, toc, loading }
}
