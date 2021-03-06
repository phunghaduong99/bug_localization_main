/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is Mozilla Communicator client code, released March 31, 1998.
 *
 * The Initial Developer of the Original Code is
 * Netscape Communications Corporation.
 * Portions created by Netscape are Copyright (C) 1998-1999
 * Netscape Communications Corporation.  All Rights Reserved.
 *
 * Contributor(s):
 *
 * IBM
 * -  Binding to permit interfacing between Mozilla and SWT
 * -  Copyright (C) 2003, 2013 IBM Corp.  All Rights Reserved.
 *
 * ***** END LICENSE BLOCK ***** */
package org.eclipse.swt.internal.mozilla;


public class nsIURI extends nsISupports {
	
	static final int LAST_METHOD_ID = nsISupports.LAST_METHOD_ID + ((IsXULRunner10() || IsXULRunner24() || IsXULRunner31()) ? 32 : 26);

	static final String NS_IURI_IID_STR = "07a22cc0-0ce5-11d3-9331-00104ba0fd40";
	static final String NS_IURI_10_IID_STR = "395fe045-7d18-4adb-a3fd-af98c8a1af11";

	static {
		IIDStore.RegisterIID(nsIURI.class, MozillaVersion.VERSION_BASE, new nsID(NS_IURI_IID_STR));
		IIDStore.RegisterIID(nsIURI.class, MozillaVersion.VERSION_XR10, new nsID(NS_IURI_10_IID_STR));
	}

	public nsIURI(long /*int*/ address) {
		super(address);
	}

	public int GetSpec(long /*int*/ aSpec) {
		return XPCOM.VtblCall(nsISupports.LAST_METHOD_ID + 1, getAddress(), aSpec);
	}

	public int GetHost(long /*int*/ aHost) {
		return XPCOM.VtblCall(nsISupports.LAST_METHOD_ID + 14, getAddress(), aHost);
	}

	public int GetPath(long /*int*/ aPath) {
		return XPCOM.VtblCall(nsISupports.LAST_METHOD_ID + 18, getAddress(), aPath);
	}
}
