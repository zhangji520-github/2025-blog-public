const fs = require('node:fs')
const path = require('node:path')
const Module = require('node:module')
const assert = require('node:assert/strict')
const ts = require('typescript')
const { htmlToDOM } = require('html-react-parser')
const filename = path.resolve('src/lib/markdown-renderer.ts')
const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const moduleUnderTest = new Module(filename, module)
moduleUnderTest.filename = filename
moduleUnderTest.paths = Module._nodeModulePaths(path.dirname(filename))
moduleUnderTest._compile(compiled, filename)
const { renderMarkdown } = moduleUnderTest.exports
const text = node => node.type === 'text' ? node.data : (node.children || []).map(text).join('')
function check(html, expected) {
  const nodes = htmlToDOM(html)
  const blocks = []
  const visit = node => { if (node.name === 'pre') blocks.push(node); (node.children || []).forEach(visit) }
  nodes.forEach(visit)
  assert.equal(blocks.length, expected.length, 'one pre per code block, without nested pre')
  blocks.forEach((block, i) => {
    assert.equal(block.attribs['data-code'], expected[i], 'copy payload preserves source')
    assert.equal(text(block).trimEnd(), expected[i].trimEnd(), 'visible code preserves text and newlines')
    assert.equal(block.children[0].name, 'code')
  })
}
async function main() {
  const a = '#include <cstdio>\nint a = 1 < 2;\n  // & "quoted" __CODE_BLOCK_0__'
  const b = '<script>alert("example")</script>\n  *ptr = x * y;'
  const sources = ['```C++\n' + a + '\n```', '```unrecognized-language\n' + b + '\n```']
  const results = await Promise.all(sources.map(renderMarkdown))
  check(results[0].html, [a]); check(results[1].html, [b])
  assert.ok(!results[1].html.includes('<script>'))
  check((await renderMarkdown('> ```Plain Text\n> line one\n>   line two\n> ```')).html, ['line one\n  line two'])
  check((await renderMarkdown('- example\n\n  ```cpp\n  int x = 1;\n  int y = 2;\n  ```')).html, ['int x = 1;\nint y = 2;'])
  const article = fs.readFileSync('public/blogs/cuda/index.md', 'utf8')
  const { marked } = require('marked')
  const expected = []
  marked.walkTokens(marked.lexer(article), t => { if (t.type === 'code') expected.push(t.text) })
  check((await renderMarkdown(article)).html, expected)
  console.log('PASS: concurrent renders, unknown language, nested blocks, copy text, and all article code blocks')
}
main().catch(error => { console.error(error); process.exitCode = 1 })
