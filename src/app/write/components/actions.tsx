import { motion } from 'motion/react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { useRouter } from 'next/navigation'
import { useWriteStore } from '../stores/write-store'
import { usePreviewStore } from '../stores/preview-store'
import { usePublish } from '../hooks/use-publish'
import JSZip from 'jszip'

const IMAGE_TYPES: Record<string, string> = {
	'.apng': 'image/apng',
	'.avif': 'image/avif',
	'.gif': 'image/gif',
	'.jpeg': 'image/jpeg',
	'.jpg': 'image/jpeg',
	'.png': 'image/png',
	'.svg': 'image/svg+xml',
	'.webp': 'image/webp'
}

const normalizeArchivePath = (value: string) => {
	let path = value.trim().replace(/^<|>$/g, '').replace(/\\/g, '/').split(/[?#]/)[0]
	try {
		path = decodeURIComponent(path)
	} catch {
		// Keep the original path when a source contains malformed escaping.
	}
	const parts = path.split('/')
	const result: string[] = []
	for (const part of parts) {
		if (!part || part === '.') continue
		if (part === '..') result.pop()
		else result.push(part)
	}
	return result.join('/')
}

const resolveArchiveImagePath = (reference: string, markdownPath: string) => {
	const normalized = normalizeArchivePath(reference)
	if (/^(?:https?:|data:|blob:|#)/i.test(reference.trim())) return null
	if (reference.trim().startsWith('/')) return normalized
	const directory = normalizeArchivePath(markdownPath).split('/').slice(0, -1)
	return normalizeArchivePath([...directory, normalized].join('/'))
}

const extensionFor = (path: string) => {
	const match = path.toLowerCase().match(/\.[a-z0-9]+$/)
	return match ? match[0] : ''
}

export function WriteActions() {
	const { loading, mode, form, loadBlogForEdit, originalSlug, updateForm, addFiles } = useWriteStore()
	const { openPreview } = usePreviewStore()
	const { isAuth, onChoosePrivateKey, onPublish, onDelete } = usePublish()
	const [saving, setSaving] = useState(false)
	const keyInputRef = useRef<HTMLInputElement>(null)
	const mdInputRef = useRef<HTMLInputElement>(null)
	const archiveInputRef = useRef<HTMLInputElement>(null)
	const router = useRouter()

	const handleImportOrPublish = () => {
		if (!isAuth) {
			keyInputRef.current?.click()
		} else {
			onPublish()
		}
	}

	const handleCancel = () => {
		if (!window.confirm('放弃本次修改吗？')) {
			return
		}
		if (mode === 'edit' && originalSlug) {
			router.push(`/blog/${originalSlug}`)
		} else {
			router.push('/')
		}
	}

	const buttonText = isAuth ? (mode === 'edit' ? '更新' : '发布') : '导入密钥'

	const handleDelete = () => {
		if (!isAuth) {
			toast.info('请先导入密钥')
			return
		}
		const confirmMsg = form?.title ? `确定删除《${form.title}》吗？该操作不可恢复。` : '确定删除当前文章吗？该操作不可恢复。'
		if (window.confirm(confirmMsg)) {
			onDelete()
		}
	}

	const handleImportMd = () => {
		mdInputRef.current?.click()
	}

	const handleImportArchive = () => {
		archiveInputRef.current?.click()
	}

	const handleMdFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0]
		if (!file) return

		try {
			const text = await file.text()
			updateForm({ md: text })
			toast.success('已导入 Markdown 文件')
		} catch (error) {
			toast.error('导入失败，请重试')
		} finally {
			if (e.currentTarget) e.currentTarget.value = ''
		}
	}

	const handleArchiveChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0]
		if (!file) return

		try {
			const zip = await JSZip.loadAsync(file)
			const markdownEntries = Object.values(zip.files).filter(entry => !entry.dir && /\.md$/i.test(entry.name))
			if (markdownEntries.length === 0) throw new Error('压缩包中没有 Markdown 文件')
			if (markdownEntries.length > 1) toast.info(`检测到 ${markdownEntries.length} 个 Markdown，已使用第一个`)

			const markdownEntry = markdownEntries[0]
			let markdown = await markdownEntry.async('text')
			const imageEntries = Object.values(zip.files).filter(entry => !entry.dir && Boolean(IMAGE_TYPES[extensionFor(entry.name)]))
			const imageFiles = await Promise.all(
				imageEntries.map(async entry => {
					const blob = await entry.async('blob')
					const ext = extensionFor(entry.name)
					return new File([blob], entry.name.split('/').pop() || `image${ext}`, { type: IMAGE_TYPES[ext] })
				})
			)
			const importedImages = await addFiles(imageFiles)
			const imagePathMap = new Map<string, string>()
			imageEntries.forEach((entry, index) => {
				const image = importedImages[index]
				if (!image) return
				imagePathMap.set(normalizeArchivePath(entry.name), `local-image:${image.id}`)
			})

			const basenameMatches = new Map<string, string[]>()
			for (const [path, placeholder] of imagePathMap) {
				const basename = path.split('/').pop() || path
				basenameMatches.set(basename, [...(basenameMatches.get(basename) || []), placeholder])
			}
			markdown = markdown.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, rawReference) => {
				const resolvedPath = resolveArchiveImagePath(rawReference, markdownEntry.name)
				if (!resolvedPath) return match
				const placeholder = imagePathMap.get(resolvedPath) || (basenameMatches.get(resolvedPath.split('/').pop() || '') || []).length === 1
					? imagePathMap.get(resolvedPath) || basenameMatches.get(resolvedPath.split('/').pop() || '')?.[0]
					: undefined
				return placeholder ? `![${alt}](${placeholder})` : match
			})

			const heading = markdown.match(/^\uFEFF?\s*#\s+([^\r\n]+)\r?\n/)
			const importedTitle = heading?.[1]?.trim()
			const suggestedSlug = (file.name.replace(/\.zip$/i, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'new-article')
			updateForm({ md: markdown, ...(importedTitle && !form.title ? { title: importedTitle } : {}), ...(!form.slug ? { slug: suggestedSlug } : {}) })
			toast.success(`已导入压缩包：Markdown 1 个，图片 ${importedImages.length} 张`)
		} catch (error) {
			console.error('Failed to import Markdown archive:', error)
			toast.error(error instanceof Error ? error.message : '压缩包导入失败，请检查文件结构')
		} finally {
			if (e.currentTarget) e.currentTarget.value = ''
		}
	}

	return (
		<>
			<input
				ref={keyInputRef}
				type='file'
				accept='.pem'
				className='hidden'
				onChange={async e => {
					const f = e.target.files?.[0]
					if (f) await onChoosePrivateKey(f)
					if (e.currentTarget) e.currentTarget.value = ''
				}}
			/>
			<input ref={mdInputRef} type='file' accept='.md' className='hidden' onChange={handleMdFileChange} />
			<input ref={archiveInputRef} type='file' accept='.zip,application/zip' className='hidden' onChange={handleArchiveChange} />

			<ul className='absolute top-4 right-6 flex items-center gap-2'>
				{mode === 'edit' && (
					<>
						<motion.div initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} className='flex items-center gap-2'>
							<div className='rounded-lg border bg-blue-50 px-4 py-2 text-sm text-blue-700'>编辑模式</div>
						</motion.div>

						<motion.button
							initial={{ opacity: 0, scale: 0.6 }}
							animate={{ opacity: 1, scale: 1 }}
							whileHover={{ scale: 1.05 }}
							whileTap={{ scale: 0.95 }}
							className='rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-600 transition-colors hover:bg-red-100'
							disabled={loading}
							onClick={handleDelete}>
							删除
						</motion.button>

						<motion.button
							whileHover={{ scale: 1.05 }}
							whileTap={{ scale: 0.95 }}
							onClick={handleCancel}
							disabled={saving}
							className='bg-card rounded-xl border px-4 py-2 text-sm'>
							取消
						</motion.button>
					</>
				)}

				<motion.button
					initial={{ opacity: 0, scale: 0.6 }}
					animate={{ opacity: 1, scale: 1 }}
					whileHover={{ scale: 1.05 }}
					whileTap={{ scale: 0.95 }}
					className='bg-card rounded-xl border px-4 py-2 text-sm'
					disabled={loading}
					onClick={handleImportMd}>
					导入 MD
				</motion.button>
				<motion.button
					initial={{ opacity: 0, scale: 0.6 }}
					animate={{ opacity: 1, scale: 1 }}
					whileHover={{ scale: 1.05 }}
					whileTap={{ scale: 0.95 }}
					className='bg-card rounded-xl border px-4 py-2 text-sm'
					disabled={loading}
					onClick={handleImportArchive}>
					导入 MD 压缩包
				</motion.button>
				<motion.button
					initial={{ opacity: 0, scale: 0.6 }}
					animate={{ opacity: 1, scale: 1 }}
					whileHover={{ scale: 1.05 }}
					whileTap={{ scale: 0.95 }}
					className='bg-card rounded-xl border px-6 py-2 text-sm'
					disabled={loading}
					onClick={openPreview}>
					预览
				</motion.button>
				<motion.button
					initial={{ opacity: 0, scale: 0.6 }}
					animate={{ opacity: 1, scale: 1 }}
					whileHover={{ scale: 1.05 }}
					whileTap={{ scale: 0.95 }}
					className='brand-btn px-6'
					disabled={loading}
					onClick={handleImportOrPublish}>
					{buttonText}
				</motion.button>
			</ul>
		</>
	)
}
