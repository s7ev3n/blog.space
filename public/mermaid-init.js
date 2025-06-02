document.addEventListener('DOMContentLoaded', function() {
  if (window.mermaid) {
    // 预先保存所有图表的源代码
    document.querySelectorAll('.mermaid').forEach(function(element) {
      if (!element.dataset.source && element.textContent) {
        element.dataset.source = element.textContent;
      }
    });
    
    // 初始化 mermaid 配置
    window.mermaid.initialize({
      startOnLoad: true,
      theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default',
      securityLevel: 'loose',
      fontFamily: 'monospace',
      flowchart: {
        curve: 'basis',
        htmlLabels: true
      },
      sequence: {
        diagramMarginX: 50,
        diagramMarginY: 10,
      },
    });
    
    // 监听主题变化
    document.addEventListener('theme-change', function() {
      const isDark = document.documentElement.classList.contains('dark');
      
      // 更新主题配置
      window.mermaid.initialize({
        theme: isDark ? 'dark' : 'default'
      });
      
      // 延时一下再重新渲染，确保DOM已经更新
      setTimeout(function() {
        // 标记所有图表为未处理
        const diagrams = document.querySelectorAll('.mermaid');
        
        // 删除所有已生成的图表内容，但保留源码
        diagrams.forEach(function(element) {
          // 保存原始内容（如果还未保存）
          if (!element.dataset.source && element.textContent) {
            element.dataset.source = element.textContent;
          }
          
          // 移除处理标记和内部内容
          if (element.hasAttribute('data-processed')) {
            element.removeAttribute('data-processed');
            
            // 恢复源代码
            if (element.dataset.source) {
              element.innerHTML = element.dataset.source;
            }
          }
        });
        
        // 重新渲染全部图表
        try {
          window.mermaid.run().catch(function(error) {
            console.error('Mermaid rendering error:', error);
            
            // 如果运行失败，再尝试一次
            setTimeout(function() {
              try {
                // 再次尝试渲染
                window.mermaid.run();
              } catch (retryError) {
                console.error('Retry mermaid render failed:', retryError);
              }
            }, 200);
          });
        } catch (e) {
          console.error('Mermaid re-render error:', e);
        }
      }, 100); // 增加延时，确保主题完全切换
    });
  }
});

// 支持 Astro View Transitions
document.addEventListener('astro:page-load', function() {
  if (window.mermaid) {
    try {
      // 确保所有新加载的图表都能被渲染
      const diagrams = document.querySelectorAll('.mermaid:not([data-processed])');
      if (diagrams.length > 0) {
        // 对于每个未处理的图表，保存原始源码
        diagrams.forEach(function(element) {
          if (!element.dataset.source && element.textContent) {
            element.dataset.source = element.textContent;
          }
        });
        
        // 运行 mermaid 渲染
        window.mermaid.run();
      }
    } catch (e) {
      console.error('Mermaid render error during page transition:', e);
    }
  }
});
