import React, { useState } from "react";
import axios from "axios";

const App: React.FC = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{ role: string; text: string }[]>([]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await axios.post("http://localhost:8000/query", {
        question: input,
      });

      const botMessage = { role: "bot", text: response.data.answer };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Error getting response from server." },
      ]);
    }

    setInput("");
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-2xl mx-auto bg-white rounded-xl shadow-md p-4">
        <h1 className="text-2xl font-bold mb-4 text-center">RAG Support Chatbot</h1>
        <div className="space-y-3 mb-4 h-96 overflow-y-auto">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`p-3 rounded-md ${msg.role === "user" ? "bg-yellow-100 text-right" : "bg-green-100 text-left"}`}
            >
              <p>{msg.text}</p>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 border p-2 rounded-md"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a support question..."
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            onClick={handleSend}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;