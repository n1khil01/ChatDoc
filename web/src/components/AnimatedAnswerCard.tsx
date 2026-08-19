import { useEffect, useRef, useState } from 'react'
import { Check } from './icons'

/* Looping typewriter demo for the landing page's "verified answer" card. Purely
   decorative -- nothing here is interactive or wired to the real API, it's a script
   replaying the same fixed example on a timer so the hero shows the product's
   behaviour instead of just describing it.

   The card frame itself (border, background, "Verified answer" header, "grounded"
   badge) is static and never animates -- only the content inside it (question text,
   figure, citation chips, verify line) types in and resets on a loop. */

const QUESTION = 'What were total net sales in 2018?'
const FIGURE = '$32,765M'
const CHIPS = ['p.58', 'p.60']

const TYPE_MS_PER_CHAR = 32
const PAUSE_AFTER_TYPE = 400
const CHIP_STAGGER = 220
const VERIFY_DELAY = 350
const HOLD_MS = 2600
const RESET_PAUSE = 550

type Stage = 'typing' | 'figure' | 'chip0' | 'chip1' | 'verify' | 'hold'

export function AnimatedAnswerCard() {
  const [typed, setTyped] = useState('')
  const [stage, setStage] = useState<Stage>('typing')
  const [cycle, setCycle] = useState(0)
  const reducedMotion = useRef(
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )

  useEffect(() => {
    if (reducedMotion.current) {
      // Skip the animation loop entirely -- show the finished state once, statically.
      setTyped(QUESTION)
      setStage('hold')
      return
    }

    let cancelled = false
    const timers: ReturnType<typeof setTimeout>[] = []
    const after = (ms: number, fn: () => void) => {
      const t = setTimeout(() => {
        if (!cancelled) fn()
      }, ms)
      timers.push(t)
    }

    function runCycle() {
      setTyped('')
      setStage('typing')
      // Remounting the reveal block (via the `cycle` key) drops the previous
      // figure/chips/verify line instantly instead of fading them out, so
      // nothing lingers on screen -- only the fade-in on the way back is animated.
      setCycle((c) => c + 1)

      for (let i = 1; i <= QUESTION.length; i++) {
        after(i * TYPE_MS_PER_CHAR, () => setTyped(QUESTION.slice(0, i)))
      }
      const typeDone = QUESTION.length * TYPE_MS_PER_CHAR

      after(typeDone + PAUSE_AFTER_TYPE, () => setStage('figure'))
      after(typeDone + PAUSE_AFTER_TYPE + CHIP_STAGGER, () => setStage('chip0'))
      after(typeDone + PAUSE_AFTER_TYPE + CHIP_STAGGER * 2, () => setStage('chip1'))
      after(typeDone + PAUSE_AFTER_TYPE + CHIP_STAGGER * 2 + VERIFY_DELAY, () => setStage('verify'))

      const revealDone = typeDone + PAUSE_AFTER_TYPE + CHIP_STAGGER * 2 + VERIFY_DELAY
      // Content-only reset: clearing reveal state (without touching the card frame)
      // lets the figure/chips/verify line fade out together in one transition
      // (all reveal booleans flip false in the same tick below) while the
      // question retypes, rather than the whole box blinking.
      after(revealDone + HOLD_MS + RESET_PAUSE, runCycle)
    }

    runCycle()
    return () => {
      cancelled = true
      timers.forEach(clearTimeout)
    }
  }, [])

  const revealed = stage !== 'typing'
  const showFigure = revealed
  const showChip0 = stage === 'chip0' || stage === 'chip1' || stage === 'verify' || stage === 'hold'
  const showChip1 = stage === 'chip1' || stage === 'verify' || stage === 'hold'
  const showVerify = stage === 'verify' || stage === 'hold'

  return (
    <article className="card demo-card" aria-hidden="true">
      <header className="demo-head">
        <span className="demo-label">Verified answer</span>
      </header>

      <p className="demo-question">
        “{typed}
        {!revealed && <span className="typing-cursor" />}”
      </p>

      <p key={`figure-${cycle}`} className={`demo-figure reveal-item${showFigure ? ' reveal-item-in' : ''}`}>
        {FIGURE}
      </p>

      <div className="demo-chips">
        {CHIPS.map((chip, i) => (
          <span
            key={`${chip}-${cycle}`}
            className={`citation-chip reveal-item${
              (i === 0 ? showChip0 : showChip1) ? ' reveal-item-in' : ''
            }`}
          >
            {chip}
          </span>
        ))}
      </div>

      <p key={`verify-${cycle}`} className={`demo-verify reveal-item${showVerify ? ' reveal-item-in' : ''}`}>
        <Check />
        Every figure matched to the chunk it cites
      </p>
    </article>
  )
}
