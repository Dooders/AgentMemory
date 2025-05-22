(() => {
  try {
    console.log('Initializing React app...');
    const { createElement, useState, useEffect } = React;
    const html = htm.bind(createElement);

    function App() {
      const [memories, setMemories] = useState([]);
      const [selected, setSelected] = useState(null);
      const [loading, setLoading] = useState(true);
      const [error, setError] = useState(null);

      // Load memories on startup
      useEffect(() => {
        loadDefaultMemories();
      }, []);

      const loadDefaultMemories = async () => {
        try {
          console.log('Starting to load default memories...');
          setLoading(true);
          setError(null);
          const result = await window.electronAPI.loadDefaultMemories();
          console.log('Received result from loadDefaultMemories:', result);
          if (result.error) {
            console.error('Error loading memories:', result.error);
            setError(result.error);
            return;
          }
          processMemoryResults(result);
        } catch (err) {
          console.error('Exception in loadDefaultMemories:', err);
          setError(err.message);
        } finally {
          setLoading(false);
        }
      };

      const processMemoryResults = (result) => {
        console.log('Processing memory results:', result);
        if (result.canceled) {
          console.log('Load was canceled');
          return;
        }
        const loaded = [];
        result.contents.forEach(item => {
          if (item.error) {
            console.error(`Failed to parse ${item.path}: ${item.error}`);
            return;
          }
          console.log('Processing item data:', item.data);
          // Handle different possible data structures
          if (Array.isArray(item.data)) {
            console.log(`Adding ${item.data.length} items from array`);
            loaded.push(...item.data);
          } else if (item.data.memories && Array.isArray(item.data.memories)) {
            console.log(`Adding ${item.data.memories.length} items from memories array`);
            loaded.push(...item.data.memories);
          } else if (item.data.memory && Array.isArray(item.data.memory)) {
            console.log(`Adding ${item.data.memory.length} items from memory array`);
            loaded.push(...item.data.memory);
          } else {
            console.log('Adding single item');
            loaded.push(item.data);
          }
        });
        console.log(`Total items loaded: ${loaded.length}`);
        if (loaded.length === 0) {
          console.log('No items were loaded, data structure:', JSON.stringify(result.contents[0].data, null, 2));
        }
        // basic sort by timestamp descending if exists
        loaded.sort((a, b) => {
          if (!a.timestamp || !b.timestamp) return 0;
          return new Date(b.timestamp) - new Date(a.timestamp);
        });
        setMemories(loaded);
        if (loaded.length) setSelected(0);
      };

      const openFiles = async () => {
        try {
          setLoading(true);
          setError(null);
          const result = await window.electronAPI.openFiles();
          processMemoryResults(result);
        } catch (err) {
          setError(err.message);
        } finally {
          setLoading(false);
        }
      };

      const MemoryList = () => html`
        <div class="sidebar">
          <div class="toolbar">
            <button onclick=${openFiles}>Open JSON Files</button>
            <button onclick=${loadDefaultMemories} style=${{ marginLeft: '8px' }}>Reload</button>
            <span style=${{ marginLeft: '8px' }}>${memories.length} items</span>
          </div>
          ${loading ? html`<div style=${{ padding: '8px' }}>Loading...</div>` : ''}
          ${error ? html`<div style=${{ padding: '8px', color: 'red' }}>${error}</div>` : ''}
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

      return html`<div style=${{ display: 'flex', width: '100%' }}>
        ${MemoryList()}
        ${MemoryDetail()}
      </div>`;
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