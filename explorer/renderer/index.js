(() => {
  try {
    console.log('Initializing React app...');
    const { createElement, useState } = React;
    const html = htm.bind(createElement);

    function App() {
      const [memories, setMemories] = useState([]);
      const [selected, setSelected] = useState(null);

      const openFiles = async () => {
        try {
          const result = await window.electronAPI.openFiles();
          if (result.canceled) return;
          const loaded = [];
          result.contents.forEach(item => {
            if (item.error) {
              console.error(`Failed to parse ${item.path}: ${item.error}`);
              return;
            }
            if (Array.isArray(item.data)) {
              loaded.push(...item.data);
            } else {
              loaded.push(item.data);
            }
          });
          // basic sort by timestamp descending if exists
          loaded.sort((a, b) => {
            if (!a.timestamp || !b.timestamp) return 0;
            return new Date(b.timestamp) - new Date(a.timestamp);
          });
          setMemories(loaded);
          if (loaded.length) setSelected(0);
        } catch (err) {
          console.error('Error in openFiles:', err);
        }
      };

      const MemoryList = () => html`
        <div class="sidebar">
          <div class="toolbar">
            <button onclick=${openFiles}>Open JSON Files</button>
            <span style=${{ marginLeft: '8px' }}>${memories.length} items</span>
          </div>
          <ul>
            ${memories.map((m, idx) => html`
              <li 
                key=${idx} 
                onclick=${() => setSelected(idx)} 
                style=${{ background: selected === idx ? '#def' : 'transparent' }}
              >
                <div><strong>${m.role || 'unknown'}</strong> - ${m.timestamp || ''}</div>
                <div style=${{ fontSize: '0.9em', color: '#555' }}>${(m.content || '').slice(0, 64)}</div>
              </li>
            `)}
          </ul>
        </div>`;

      const MemoryDetail = () => {
        if (selected === null || !memories[selected])
          return html`<div class="main">No memory selected</div>`;
        const mem = memories[selected];
        return html`<div class="main">
          <h2>Memory Detail</h2>
          <pre style=${{ whiteSpace: 'pre-wrap' }}>${JSON.stringify(mem, null, 2)}</pre>
        </div>`;
      };

      return html`<>
        ${MemoryList()}
        ${MemoryDetail()}
      </>`;
    }

    console.log('Rendering React app...');
    ReactDOM.render(createElement(App), document.getElementById('root'));
    console.log('React app rendered successfully');
  } catch (err) {
    console.error('Error initializing React app:', err);
    document.body.innerHTML = `<div style="color: red; padding: 20px;">
      <h2>Error initializing app:</h2>
      <pre>${err.message}</pre>
    </div>`;
  }
})(); 