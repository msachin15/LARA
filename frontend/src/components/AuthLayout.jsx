/**
 * Guard for pages that require a signed-in user.
 *
 * While Clerk is still loading, show a spinner.
 * If there is no session, send the visitor to /sign-in.
 * Otherwise render whatever page was wrapped in <AuthLayout>.
 */
import { useUser } from "@clerk/clerk-react";
import { Navigate } from "react-router-dom";

const AuthLayout = ({ children }) => {
  const { isSignedIn, isLoaded } = useUser();

  if (!isLoaded) {
    // Same idea as App.jsx: wait until Clerk knows whether a session exists.
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (!isSignedIn) {
    // replace is not set, so the user can press Back to the page they came from.
    return <Navigate to="/sign-in" />;
  }

  // children is the protected page passed in by the parent.
  return children;
};

export default AuthLayout;