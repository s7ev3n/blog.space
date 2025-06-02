/**
 * remark-mermaid.ts
 * 
 * 这个插件将 Markdown 中的 ```mermaid 代码块转换为 HTML <div class="mermaid"> 元素
 * 这样可以与全局加载的 mermaid 库无缝配合，实现图表渲染
 */

import { visit } from "unist-util-visit";
import type { Root, Code } from "mdast";

/**
 * 将 Markdown 中的 ```mermaid 代码块转换为 <div class="mermaid"> HTML 元素
 * 与其他替代方案相比，这是目前最简洁、最可靠的实现方式
 */
export function remarkMermaid() {
  return (tree: Root) => {
    visit(tree, "code", (node) => {
      const codeNode = node as Code;
      
      if (codeNode.lang === "mermaid") {
        // 修改节点类型为 html
        const htmlNode = node as unknown as { type: string; value: string };
        htmlNode.type = "html";
        
        // 输出为 <div class="mermaid"> 元素
        htmlNode.value = `<div class="mermaid">\n${codeNode.value}\n</div>`;
      }
    });
  };
}
