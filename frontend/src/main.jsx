/**
 * App entry for the browser.
 *
 * Wraps the React tree in:
 *   ClerkProvider  — login/signup (needs VITE_CLERK_PUBLISHABLE_KEY in frontend/.env)
 *   BrowserRouter  — page URLs such as /, /sign-in, /chat
 *   App            — the actual routes
 */
import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ClerkProvider } from '@clerk/clerk-react';
import App from './App.jsx';
import './index.css';

// Clerk's public key. Safe to ship to the browser. The secret key stays on Clerk's servers.
const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

if (!PUBLISHABLE_KEY) {
  // Stop immediately so a missing .env is obvious instead of a blank auth screen.
  throw new Error("Missing Publishable Key");
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY} routing="path">
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ClerkProvider>
  </React.StrictMode>
);
