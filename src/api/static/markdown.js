// A small Markdown renderer for answer text.
//
// It builds DOM nodes and never touches innerHTML, so model output cannot inject
// markup and no sanitizer is needed. That constraint is the reason this exists
// rather than a CDN parser: the alternative is marked + DOMPurify, two more
// network dependencies on the one path where a mistake is an XSS hole.
//
// Covers what answers actually contain: headings, nested lists, fenced and inline
// code, emphasis, links, block quotes, rules. Not tables -- they would render as
// paragraphs, and the knowledge base has none.

const FENCE = /^\s*```(\w*)\s*$/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const QUOTE = /^>\s?(.*)$/;
const RULE = /^\s*(---+|\*\*\*+|___+)\s*$/;
const ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

// Anything else is a scheme we will not put in an href.
const SAFE_HREF = /^(https?:\/\/|mailto:|#|\/)/i;

const el = (tag) => document.createElement(tag);

/**
 * @param {HTMLElement} target  emptied, then filled
 * @param {string} text         markdown
 * @param {(n: string) => Node} citation  builds the chip for a `[S1]` marker
 */
export function renderMarkdown(target, text, citation) {
  target.replaceChildren(...blocks(text.split("\n"), citation));
}

function blocks(lines, citation) {
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
    } else if (FENCE.test(line)) {
      const lang = line.match(FENCE)[1];
      const body = [];
      i++;
      // An unclosed fence runs to the end -- which is what a half-streamed
      // code block looks like, so it renders correctly while still arriving.
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
      if (i < lines.length) i++;
      const code = el("code");
      code.textContent = body.join("\n");
      if (lang) code.dataset.lang = lang;
      const pre = el("pre");
      pre.append(code);
      out.push(pre);
    } else if (RULE.test(line)) {
      out.push(el("hr"));
      i++;
    } else if (HEADING.test(line)) {
      const [, hashes, content] = line.match(HEADING);
      const heading = el(`h${Math.min(hashes.length + 1, 6)}`);
      heading.append(...inline(content, citation));
      out.push(heading);
      i++;
    } else if (QUOTE.test(line)) {
      const body = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        body.push(lines[i++].match(QUOTE)[1]);
      }
      const quote = el("blockquote");
      quote.append(...blocks(body, citation));
      out.push(quote);
    } else if (ITEM.test(line)) {
      const body = [];
      // A list ends at a blank line that is not followed by more of the list,
      // so indented continuation paragraphs stay inside their item.
      while (i < lines.length) {
        if (ITEM.test(lines[i]) || /^\s+\S/.test(lines[i])) {
          body.push(lines[i++]);
        } else if (!lines[i].trim() && i + 1 < lines.length && /^\s+\S/.test(lines[i + 1])) {
          body.push(lines[i++]);
        } else {
          break;
        }
      }
      out.push(list(body, citation));
    } else {
      const body = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !FENCE.test(lines[i]) &&
        !HEADING.test(lines[i]) &&
        !QUOTE.test(lines[i]) &&
        !RULE.test(lines[i]) &&
        !ITEM.test(lines[i])
      ) {
        body.push(lines[i++]);
      }
      const paragraph = el("p");
      paragraph.append(...inline(body.join("\n"), citation));
      out.push(paragraph);
    }
  }
  return out;
}

/** One list level. Deeper indentation recurses. */
function list(lines, citation) {
  const [, indent, first] = lines[0].match(ITEM);
  const ordered = /\d/.test(first);
  const root = el(ordered ? "ol" : "ul");
  if (ordered) {
    const start = parseInt(first, 10);
    if (start !== 1) root.start = start;
  }

  let current = null;
  for (const line of lines) {
    const match = line.match(ITEM);
    if (match && match[1].length <= indent.length) {
      current = el("li");
      current.dataset.lines = match[3];
      current._buffer = [match[3]];
      root.append(current);
    } else if (current) {
      // Dedent by the item's own indent so nested levels parse from column 0.
      current._buffer.push(line.slice(indent.length + 2));
    }
  }

  for (const item of root.children) {
    const children = blocks(item._buffer, citation);
    // A single paragraph is unwrapped: <li><p>x</p></li> double-spaces a list.
    if (children.length === 1 && children[0].tagName === "P") {
      item.replaceChildren(...children[0].childNodes);
    } else {
      item.replaceChildren(...children);
    }
    delete item._buffer;
    delete item.dataset.lines;
  }
  return root;
}

const INLINE = [
  [/^`([^`\n]+)`/, (m) => text(el("code"), m[1])],
  [/^\*\*([\s\S]+?)\*\*/, (m, c) => wrap("strong", m[1], c)],
  [/^__([\s\S]+?)__/, (m, c) => wrap("strong", m[1], c)],
  [/^~~([\s\S]+?)~~/, (m, c) => wrap("del", m[1], c)],
  [/^\*([^*\n]+)\*/, (m, c) => wrap("em", m[1], c)],
  [/^_([^_\n]+)_/, (m, c) => wrap("em", m[1], c)],
  [/^\[S(\d+)\]/, (m, c) => c(m[1])],
  [/^\[([^\]\n]*)\]\(([^)\s]+)\)/, (m, c) => link(m[1], m[2], c)],
];

function inline(source, citation) {
  const out = [];
  let buffer = "";
  let i = 0;

  const flush = () => {
    if (buffer) out.push(document.createTextNode(buffer));
    buffer = "";
  };

  while (i < source.length) {
    const rest = source.slice(i);
    const previous = source[i - 1] || " ";
    let matched = false;

    for (const [re, build] of INLINE) {
      // `_` inside a word is snake_case, not emphasis.
      if (rest[0] === "_" && /\w/.test(previous)) break;
      const match = rest.match(re);
      if (!match) continue;
      flush();
      out.push(build(match, citation));
      i += match[0].length;
      matched = true;
      break;
    }

    if (!matched) {
      if (source[i] === "\n") {
        flush();
        out.push(el("br"));
      } else {
        buffer += source[i];
      }
      i++;
    }
  }
  flush();
  return out;
}

function text(node, value) {
  node.textContent = value;
  return node;
}

function wrap(tag, content, citation) {
  const node = el(tag);
  node.append(...inline(content, citation));
  return node;
}

function link(label, href, citation) {
  // Refuse javascript: and data: rather than render an unclickable-looking trap.
  if (!SAFE_HREF.test(href)) return document.createTextNode(`[${label}](${href})`);
  const anchor = el("a");
  anchor.href = href;
  anchor.target = "_blank";
  anchor.rel = "noopener noreferrer";
  anchor.append(...inline(label, citation));
  return anchor;
}
