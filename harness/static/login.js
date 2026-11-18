// The login form: POST /api/login sets an HttpOnly session cookie, then the dashboard loads.
const form = document.getElementById("login");
const error = document.getElementById("error");
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  error.textContent = "";
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const res = await fetch("/api/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: form.name.value.trim(), password: form.password.value }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "couldn't log in");
    location.href = "/";
  } catch (err) {
    error.textContent = err.message;
    form.password.value = "";
    form.password.focus();
  } finally {
    button.disabled = false;
  }
});
