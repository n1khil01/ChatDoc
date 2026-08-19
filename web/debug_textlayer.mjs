import { readFileSync } from 'fs'
import * as pdfjsLib from 'pdfjs-dist/build/pdf.mjs'

pdfjsLib.GlobalWorkerOptions.workerSrc = './node_modules/pdfjs-dist/build/pdf.worker.mjs'

const data = new Uint8Array(readFileSync('./midnight_debug.pdf'))
const doc = await pdfjsLib.getDocument({ data }).promise
const page = await doc.getPage(1)
const textContent = await page.getTextContent()

const items = textContent.items.filter((it) => typeof it.str === 'string' && it.str)
console.log('--- raw items (all, in original order) ---')
for (const it of items) {
  console.log(JSON.stringify({ str: it.str, x: it.transform[4], y: it.transform[5], width: it.width, height: it.height }))
}

function normalize(s) {
  return s.replace(/\s+/g, ' ').trim().toLowerCase()
}

const sorted = [...items].sort((a, b) => {
  const dy = b.transform[5] - a.transform[5]
  if (Math.abs(dy) > 2) return dy
  return a.transform[4] - b.transform[4]
})

let concatenated = ''
let prev = null
for (const item of sorted) {
  let sep = ''
  if (prev) {
    const sameLine = Math.abs(item.transform[5] - prev.transform[5]) <= 2
    if (!sameLine) {
      sep = ' '
    } else {
      const prevEndX = prev.transform[4] + prev.width
      const gap = item.transform[4] - prevEndX
      const avgCharWidth = prev.width / Math.max(prev.str.length, 1)
      if (gap > avgCharWidth * 0.35) sep = ' '
    }
  }
  concatenated += sep + normalize(item.str)
  prev = item
}

console.log('--- concatenated (first 300 chars) ---')
console.log(JSON.stringify(concatenated.slice(0, 300)))

const target = normalize('Midnight Calypso Capstone Project Kickoff Document')
console.log('--- search target ---', JSON.stringify(target))
console.log('--- indexOf result ---', concatenated.indexOf(target))

process.exit(0)
