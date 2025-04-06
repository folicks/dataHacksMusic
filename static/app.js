document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("lyricsForm")
    const seedInput = document.getElementById("seed")
    const lengthInput = document.getElementById("length")
    const resultDiv = document.getElementById("result")
  
    form.addEventListener("submit", async (e) => {
      e.preventDefault()
      resultDiv.textContent = "Generating..."
      document.body.classList.add("loading")
  
      try {
        const response = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            seed: seedInput.value,
            length: lengthInput.value,
          }),
        })
  
        const data = await response.json()
        if (data.error) {
          resultDiv.textContent = `Error: ${data.error}`
        } else {
          resultDiv.textContent = data.lyrics
        }
      } catch (error) {
        console.error("Error:", error)
        resultDiv.textContent = "Error generating lyrics"
      } finally {
        document.body.classList.remove("loading")
      }
    })
  })
  
  