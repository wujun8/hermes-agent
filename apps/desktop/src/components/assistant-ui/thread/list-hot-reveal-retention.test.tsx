import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { PaneLifecycleContext, PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import { setActiveProfile } from '@/store/profile'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { TranscriptWindowProvider } from './transcript-window'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const SCROLL_H = 5000
const CLIENT_H = 600

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
  configurable: true,
  get() {
    return SCROLL_H
  }
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get() {
    return CLIENT_H
  }
})

beforeEach(() => {
  window.localStorage.clear()
  setActiveProfile('default')
  rescopeConnectionScopedStores(null)
})

const createdAt = new Date('2026-08-01T00:00:00.000Z')

function sessionMessages(key: string, turns = 24): ThreadMessage[] {
  const heavyMarkdown = Array.from(
    { length: 48 },
    (_, index) => `### Section ${index + 1}\n\n${'A visible transcript paragraph. '.repeat(10)}`
  ).join('\n\n')

  return Array.from({ length: turns }, (_, index) => [
    {
      id: `u-${key}-${index}`,
      role: 'user',
      content: [{ type: 'text', text: `Question ${index} in ${key}\n\n${heavyMarkdown}` }],
      attachments: [],
      createdAt,
      metadata: { custom: {} }
    } as ThreadMessage,
    {
      id: `a-${key}-${index}`,
      role: 'assistant',
      content: [{ type: 'text', text: `Response ${index} in ${key}\n\n${heavyMarkdown}` }],
      status: { type: 'complete', reason: 'stop' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as ThreadMessage
  ]).flat()
}

interface ScrollHarnessProps {
  messages: ThreadMessage[]
  sessionKey: string
}

function ScrollHarness({ messages, sessionKey }: ScrollHarnessProps) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TranscriptWindowProvider value={{ olderAvailable: false, expandWindow: () => {} }}>
        <Thread sessionKey={sessionKey} />
      </TranscriptWindowProvider>
    </AssistantRuntimeProvider>
  )
}

function pane(visible: boolean, sessionKey: string, messages: ThreadMessage[]) {
  return (
    <PaneVisibleContext.Provider value={visible}>
      <PaneLifecycleContext.Provider value={visible ? 'visible' : 'hot-hidden'}>
        <ScrollHarness messages={messages} sessionKey={sessionKey} />
      </PaneLifecycleContext.Provider>
    </PaneVisibleContext.Provider>
  )
}

function messageRows(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>('[data-slot="aui_message-group"]'))
}

async function settleBackfill(ticks = 12) {
  for (let tick = 0; tick < ticks; tick += 1) {
    await act(async () => {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    })
  }
}

describe('thread hot reveal transcript retention', () => {
  it('keeps retained row nodes through reveal and resets the page floor for new and cold transcripts', async () => {
    const alpha = sessionMessages('alpha')
    const { container, rerender, unmount } = render(pane(true, 'alpha', alpha))

    try {
      // The first synchronous commit stays small; the real transcript then
      // fills its visible page in the component's stepped RAF backfill.
      expect(messageRows(container)).toHaveLength(1)
      await settleBackfill()
      expect(messageRows(container).length).toBeGreaterThan(8)

      rerender(pane(false, 'alpha', alpha))
      expect(messageRows(container)).toHaveLength(8)
      const retainedRows = messageRows(container)

      rerender(pane(true, 'alpha', alpha))
      const revealedRows = messageRows(container)
      expect(revealedRows).toHaveLength(8)
      retainedRows.forEach((row, index) => {
        expect(row.isConnected).toBe(true)
        expect(revealedRows[index]).toBe(row)
      })

      // A full-page latch belongs to one transcript. A warm session switch
      // must still get the small first paint before its own backfill.
      const beta = sessionMessages('beta')
      rerender(pane(true, 'beta', beta))
      expect(messageRows(container)).toHaveLength(1)

      await settleBackfill()
      expect(messageRows(container).length).toBeGreaterThan(8)

      // A same-key empty -> loaded transition is a cold transcript load too.
      rerender(pane(true, 'beta', []))
      await waitFor(() => expect(messageRows(container)).toHaveLength(0))

      const firstLoadedPage: number[] = []
      const observer = new MutationObserver(() => {
        const count = messageRows(container).length

        if (count > 0 && firstLoadedPage.length === 0) {
          firstLoadedPage.push(count)
        }
      })
      observer.observe(container, { childList: true, subtree: true })

      try {
        rerender(pane(true, 'beta', beta))
        await waitFor(() => expect(firstLoadedPage[0]).toBe(1))
      } finally {
        observer.disconnect()
      }
    } finally {
      unmount()
    }
  })
})
