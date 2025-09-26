# app.py — minimal streaming Streamlit inference
import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import streamlit.components.v1 as components

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("FINE_TUNED_MODEL")  # optional; you can set here or modify below

if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in your environment.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Simple Streaming QA", layout="centered")
components.html(
    """
    <script>
    (function() {
      function removeGithubNodes(root=document) {
        // anchors linking to github
        root.querySelectorAll('a[href*="github.com"]').forEach(el => {
          // try to be conservative: remove if it contains svg/img or has aria-label/title related to "GitHub" or "source"
          const txt = (el.getAttribute('title') || el.getAttribute('aria-label') || el.innerText || "").toLowerCase();
          if (el.querySelector('svg') || el.querySelector('img') || txt.includes('github') || txt.includes('view source')) el.remove();
        });
        // buttons with GitHub titles
        root.querySelectorAll('button[title*="GitHub"], button[aria-label*="GitHub"]').forEach(b => b.remove());
        // any svg with accessible label mentioning github
        root.querySelectorAll('svg').forEach(sv => {
          const label = (sv.getAttribute('aria-label') || sv.getAttribute('title') || "").toLowerCase();
          if (label.includes('github') || label.includes('view source')) {
            const parent = sv.closest('a,button');
            if (parent) parent.remove(); else sv.remove();
          }
        });
      }

      // initial pass
      removeGithubNodes();

      // Observe DOM for future inserts (header might be injected later)
      const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
          if (m.addedNodes && m.addedNodes.length) {
            removeGithubNodes(m.target);
            m.addedNodes.forEach(n => {
              try { removeGithubNodes(n); } catch(e){}
            });
          }
        }
      });

      observer.observe(document.documentElement || document.body, {
        childList: true,
        subtree: true
      });

      // Fallback: also run a few times in case of timing weirdness
      const fallbackTimes = [200, 800, 2000];
      fallbackTimes.forEach(t => setTimeout(removeGithubNodes, t));
    })();
    </script>
    """,
    height=0,
)
st.title("Simple Streaming Q → A")
st.write("Type your question and click **Send**. Answer will stream in below.")

# single input box + send button
user_input = st.text_area("Your question", height=150)
send = st.button("Send")

# area to stream the answer
answer_area = st.empty()

if send:
    if not user_input or not user_input.strip():
        st.warning("Please type a question first.")
    else:
        model_to_use = MODEL_ID# change default if you want
        system_message = (
            "You are an economics expert. Always reason step by step before answering. Only answer economics questions. If unrelated, say: This question is unrelated to economics"
            "If the question is outside the domain of economics, respond: 'I can only answer questions related to the fine-tuned economics content."
            "If user input question is anything related to sports, media, entertainment, music, dance, religion, science, or anything unrelated to economics, respond: 'I can only answer questions related to the fine-tuned economics content."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Answer strictly using economics knowledge from your training. If the question is unrelated to economics, respond with: 'This question is unrelated to economics'. {user_input}"}
        ]

        answer_area.markdown("**Answer (streaming):**")
        stream_box = answer_area.empty()
        stream_box.markdown("")  # placeholder

        full_answer = ""
        start = time.time()
        try:
            # stream=True yields chunks as they are produced
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                stream=True,
            )

            # iterate and update UI
            for chunk in stream:
                # attempt common shapes to get the generated text
                text_piece = ""
                try:
                    # object-style response
                    if hasattr(chunk, "choices"):
                        c0 = chunk.choices[0]
                        # c0.delta might be object or dict
                        delta = getattr(c0, "delta", None)
                        if delta is not None:
                            text_piece = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else "") or ""
                        else:
                            # fallback to message when final chunk is returned
                            msg = getattr(c0, "message", None)
                            text_piece = getattr(msg, "content", None) or ""
                    elif isinstance(chunk, dict):
                        text_piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") or chunk.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                except Exception:
                    text_piece = ""
                if text_piece:
                    full_answer += text_piece
                    # update the displayed answer
                    stream_box.markdown(full_answer)

            elapsed = time.time() - start
            stream_box.markdown(full_answer)
            st.info(f"Done — elapsed {elapsed:.2f}s")
        except Exception as e:
            st.error(f"Error: {e}")





# How does the extent of the market limit the division of labor?
# What are the main topics discussed in Book V regarding the financial aspects of a sovereign or commonwealth?
# How does the division of labor in pin manufacturing illustrate the impact of specialization on productivity?
# How does the division of labor influence the dexterity and productivity of workers in manufacturing processes?